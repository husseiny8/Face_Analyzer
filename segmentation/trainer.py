import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import (autocast,GradScaler)
from multitask_model import MultiTaskModel


class Trainer:

    def __init__(
            self,
            encoder_name="clip",
            model_path=None,
            projection_dim=512,
            freeze_encoder=True,
            lr=1e-4,
            weight_decay=1e-4,
            gender_weight=1.0,
            age_weight=1.0,
            race_weight=1.0,
            checkpoint_dir="checkpoints",
            patience=5
    ):

        self.device = torch.device("cuda"if torch.cuda.is_available()else "cpu")

        print("=" * 60)
        print(f"Device : {self.device}")
        print("=" * 60)


        # Model
        self.model = MultiTaskModel(
            encoder_name=encoder_name,
            model_path=model_path,
            projection_dim=projection_dim,
            freeze_encoder=freeze_encoder
        )

        self.model.to(self.device)


        # Loss Functions
        self.gender_criterion = nn.BCEWithLogitsLoss()
        self.age_criterion = nn.CrossEntropyLoss()
        self.race_criterion = nn.CrossEntropyLoss()


        # Loss Weights
        self.gender_weight = gender_weight
        self.age_weight = age_weight
        self.race_weight = race_weight


        # Optimizer
        self.optimizer = AdamW(
            filter(
                lambda p: p.requires_grad,self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay

        )

        # Learning Rate Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )

        # Mixed Precision
        self.scaler = GradScaler(enabled=torch.cuda.is_available())


        # Checkpoint Directory
        self.checkpoint_dir = Path(checkpoint_dir)

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Early Stopping
        self.best_validation_loss = float("inf")
        self.early_stop_counter = 0
        self.patience = patience

        # History
        self.history = {
            "train_loss": [],
            "validation_loss": [],
            "gender_loss": [],
            "age_loss": [],
            "race_loss": [],
            "gender_accuracy": [],
            "age_accuracy": [],
            "race_accuracy": [],
            "learning_rate": []
        }

    # Save Checkpoint
    def save_checkpoint(self,filename="best_model.pt"):

        checkpoint = {
            "model_state_dict":self.model.state_dict(),
            "optimizer_state_dict":self.optimizer.state_dict(),
            "scheduler_state_dict":self.scheduler.state_dict(),
            "history":self.history,
            "best_validation_loss":self.best_validation_loss
        }

        torch.save(checkpoint,self.checkpoint_dir / filename)

    # Load Checkpoint
    def load_checkpoint(self,filename="best_model.pt"):

        checkpoint = torch.load(
            self.checkpoint_dir / filename,
            map_location=self.device
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        self.scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

        self.history = checkpoint["history"]

        self.best_validation_loss = checkpoint[
            "best_validation_loss"
        ]

        print(f"Checkpoint Loaded : {filename}")

    # Train One Epoch
    def train_epoch(self, train_loader):

        self.model.train()

        running_loss = 0.0
        running_gender_loss = 0.0
        running_age_loss = 0.0
        running_race_loss = 0.0
        gender_correct = 0
        age_correct = 0
        race_correct = 0
        total_samples = 0

        progress = tqdm(
            train_loader,
            desc="Training",
            leave=False
        )

        for batch in progress:

            # Move Batch To Device
            images = batch["pixel_values"].to(self.device)
            gender_labels = batch["gender"].to(self.device).unsqueeze(1)
            age_labels = batch["age"].to(self.device)
            race_labels = batch["race"].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            with autocast(enabled=torch.cuda.is_available()):

                outputs = self.model(images)

                # Losses
                gender_loss = self.gender_criterion(
                    outputs["gender"],
                    gender_labels
                )

                age_loss = self.age_criterion(
                    outputs["age"],
                    age_labels
                )

                race_loss = self.race_criterion(
                    outputs["race"],
                    race_labels
                )


                loss = (
                    self.gender_weight * gender_loss +
                    self.age_weight * age_loss +
                    self.race_weight * race_loss
                )

            # Backpropagation
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Statistics
            batch_size = images.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size
            running_gender_loss += gender_loss.item() * batch_size
            running_age_loss += age_loss.item() * batch_size
            running_race_loss += race_loss.item() * batch_size

            # Gender Accuracy
            gender_prediction = (torch.sigmoid(outputs["gender"]) >= 0.5).float()

            gender_correct += (gender_prediction == gender_labels).sum().item()

            # Age Accuracy
            age_prediction = outputs["age"].argmax(dim=1)

            age_correct += (age_prediction == age_labels).sum().item()

            # Race Accuracy
            race_prediction = outputs["race"].argmax(dim=1)

            race_correct += (race_prediction == race_labels).sum().item()

            # Progress Bar
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                gender=f"{gender_loss.item():.4f}",
                age=f"{age_loss.item():.4f}",
                race=f"{race_loss.item():.4f}"
            )

        # Epoch Statistics
        epoch_loss = running_loss / total_samples
        gender_loss = running_gender_loss / total_samples
        age_loss = running_age_loss / total_samples
        race_loss = running_race_loss / total_samples
        gender_accuracy = 100.0 * gender_correct / total_samples
        age_accuracy = 100.0 * age_correct / total_samples
        race_accuracy = 100.0 * race_correct / total_samples

        return {
            "loss": epoch_loss,
            "gender_loss": gender_loss,
            "age_loss": age_loss,
            "race_loss": race_loss,
            "gender_accuracy": gender_accuracy,
            "age_accuracy": age_accuracy,
            "race_accuracy": race_accuracy
        }

    # Validation
    @torch.no_grad()
    def validate_epoch(self, validation_loader):

        self.model.eval()
        running_loss = 0.0
        running_gender_loss = 0.0
        running_age_loss = 0.0
        running_race_loss = 0.0
        gender_correct = 0
        age_correct = 0
        race_correct = 0
        total_samples = 0

        progress = tqdm(validation_loader,desc="Validation",leave=False)

        for batch in progress:

            images = batch["pixel_values"].to(self.device)
            gender_labels = batch["gender"].to(self.device).unsqueeze(1)
            age_labels = batch["age"].to(self.device)
            race_labels = batch["race"].to(self.device)

            outputs = self.model(images)

            gender_loss = self.gender_criterion(outputs["gender"],gender_labels)

            age_loss = self.age_criterion(outputs["age"],age_labels)

            race_loss = self.race_criterion(outputs["race"],race_labels)


            loss = (
                self.gender_weight * gender_loss +
                self.age_weight * age_loss +
                self.race_weight * race_loss
            )


            batch_size = images.size(0)
            total_samples += batch_size
            running_loss += loss.item() * batch_size
            running_gender_loss += gender_loss.item() * batch_size
            running_age_loss += age_loss.item() * batch_size
            running_race_loss += race_loss.item() * batch_size


            gender_prediction = (torch.sigmoid(outputs["gender"]) >= 0.5).float()

            gender_correct += (gender_prediction == gender_labels).sum().item()

            age_prediction = outputs["age"].argmax(dim=1)

            age_correct += (age_prediction == age_labels).sum().item()


            race_prediction = outputs["race"].argmax(dim=1)

            race_correct += (race_prediction == race_labels).sum().item()


        epoch_loss = running_loss / total_samples
        gender_loss = running_gender_loss / total_samples
        age_loss = running_age_loss / total_samples
        race_loss = running_race_loss / total_samples
        gender_accuracy = 100.0 * gender_correct / total_samples
        age_accuracy = 100.0 * age_correct / total_samples
        race_accuracy = 100.0 * race_correct / total_samples

        return {
            "loss": epoch_loss,
            "gender_loss": gender_loss,
            "age_loss": age_loss,
            "race_loss": race_loss,
            "gender_accuracy": gender_accuracy,
            "age_accuracy": age_accuracy,
            "race_accuracy": race_accuracy
        }

    def fit(self,train_loader,validation_loader,epochs):

        print("=" * 70)
        print("Training Started")
        print("=" * 70)

        for epoch in range(epochs):

            print()
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 70)


            # Train
            train_result = self.train_epoch(train_loader)

            # Validation
            validation_result = self.validate_epoch(
                validation_loader
            )

            # Scheduler
            self.scheduler.step(validation_result["loss"])

            # Save History
            self.history["train_loss"].append(train_result["loss"])

            self.history["validation_loss"].append(validation_result["loss"])

            self.history["gender_loss"].append(validation_result["gender_loss"])

            self.history["age_loss"].append(validation_result["age_loss"])

            self.history["race_loss"].append(validation_result["race_loss"])

            self.history["gender_accuracy"].append(validation_result["gender_accuracy"])

            self.history["age_accuracy"].append(validation_result["age_accuracy"])

            self.history["race_accuracy"].append(validation_result["race_accuracy"])

            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

            # Print Results
            print(f"Train Loss      : {train_result['loss']:.4f}")

            print(f"Validation Loss : {validation_result['loss']:.4f}")

            print()

            print(f"Gender Accuracy : {validation_result['gender_accuracy']:.2f}%")

            print(f"Age Accuracy    : {validation_result['age_accuracy']:.2f}%")

            print(f"Race Accuracy   : {validation_result['race_accuracy']:.2f}%")

            print()

            print(f"Learning Rate   : {self.optimizer.param_groups[0]['lr']:.2e}")

            # Best Model
            if validation_result["loss"] < self.best_validation_loss:
                self.best_validation_loss = validation_result["loss"]
                self.early_stop_counter = 0
                self.save_checkpoint("best_model.pt")

                print()
                print("Best Model Saved.")

            else:

                self.early_stop_counter += 1
                print()
                print(
                    f"EarlyStopping Counter : "
                    f"{self.early_stop_counter}/{self.patience}")

            # Early Stop
            if self.early_stop_counter >= self.patience:
                print()
                print("=" * 60)
                print("Early Stopping Activated")
                print("=" * 60)
                break

        # Save Final Model
        self.save_checkpoint("last_model.pt")

        # Save History
        history = pd.DataFrame(self.history)

        history.to_csv(self.checkpoint_dir / "history.csv",index=False)


        print()
        print("=" * 70)
        print("Training Finished")
        print("=" * 70)

        return history

    # Predict
    @torch.no_grad()
    def predict(self,dataloader):

        self.model.eval()
        predictions = []

        progress = tqdm(dataloader,desc="Predict")

        for batch in progress:

            images = batch["pixel_values"].to(self.device)

            outputs = self.model.predict(images)

            predictions.append(
                {
                    "gender":
                        outputs["gender_prediction"].cpu(),

                    "age":
                        outputs["age_prediction"].cpu(),

                    "race":
                        outputs["race_prediction"].cpu()
                }
            )

        return predictions

    # Save Model Only
    def save_model(self,filename="model.pt"):

        torch.save(
            self.model.state_dict(),
            self.checkpoint_dir / filename
        )

    # Load Model Only
    def load_model(self,filename="model.pt"):

        self.model.load_state_dict(
            torch.load(
                self.checkpoint_dir / filename,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

        print(f"Model Loaded : {filename}")