import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import albumentations as A

# ============================
# ENABLE MIXED PRECISION
# This was done to reduce the computational load and memory usage due to hardware restrictions.
# ============================
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
print("⚡ Mixed precision enabled.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ========== GPU CONFIG ==========
# The model needed to be trained on GPU due to its size and complexity.
# This section ensures that TensorFlow uses the GPU if available.
# Running this requires the installation of the appropriate GPU drivers and CUDA toolkit from NVIDIA.
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    raise SystemExit("❌ No GPU detected — aborting training.")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print(f"✅ {len(gpus)} GPU(s) detected")

# ========== CONFIG ==========
# Hyperparameters
IMG_SIZE = (768, 1024)
BATCH_SIZE = 6
EPOCHS = 25

# Class balancing parameters
# To address class imbalance, we undersample no-droplet images.
BALANCE_CLASSES = True
NO_DROPLET_SAMPLE_RATE = 0.3  # Back to what worked (got 87% dice)

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints_stable")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================
# Data Augmentation
# ============================
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussianBlur(blur_limit=(3, 5), p=0.3),
    A.RandomGamma(gamma_limit=(80, 120), p=0.3),
])

# ============================
# Load Image Paths
# Locates all images in the train directory and matches them with their masks.
# ============================
IMAGE_ROOT = r"C:/Users/peter/Desktop/Courses/Machine Learning/project/RaindropsOnWindshield/images"
MASK_ROOT = r"C:/Users/peter/Desktop/Courses/Machine Learning/project/RaindropsOnWindshield/masks"

image_paths, mask_paths = [], []
for root, _, files in os.walk(IMAGE_ROOT):
    for filename in sorted(files):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(root, filename)
            rel = os.path.relpath(img_path, IMAGE_ROOT)
            mask_path = os.path.join(MASK_ROOT, rel)
            if os.path.exists(mask_path):
                image_paths.append(img_path)
                mask_paths.append(mask_path)

print(f"Found {len(image_paths)} valid image/mask pairs.")

# ============================
# APPLY CLASS BALANCING
# ============================
if BALANCE_CLASSES:
    balanced_img, balanced_mask = [], []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        mask = load_img(mask_path, color_mode="grayscale")
        mask_array = img_to_array(mask)
        has_droplets = np.mean(mask_array) > 1
        
        if has_droplets:
            balanced_img.append(img_path)
            balanced_mask.append(mask_path)
        else:
            if np.random.random() < NO_DROPLET_SAMPLE_RATE:
                balanced_img.append(img_path)
                balanced_mask.append(mask_path)
    
    image_paths, mask_paths = balanced_img, balanced_mask
    print(f"✅ After class balancing: {len(image_paths)} images")
    
    droplet_count = sum(1 for m in mask_paths if np.mean(img_to_array(load_img(m, color_mode="grayscale"))) > 1)
    print(f"   With droplets: {droplet_count}, Without: {len(image_paths) - droplet_count}")

# ============================
# 80-20 Train-Test Split
# ============================
train_img, val_img, train_mask, val_mask = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

print(f"Training: {len(train_img)}, Validation: {len(val_img)}")

# ============================
# Dataset pipeline
# Apply preprocessing and augmentation
# ============================
def load_and_preprocess(img_path, mask_path, augment=False):
    img = load_img(img_path, target_size=IMG_SIZE)
    mask = load_img(mask_path, target_size=IMG_SIZE, color_mode="grayscale")
    
    img = img_to_array(img).astype(np.uint8)
    mask = img_to_array(mask).astype(np.uint8)
    
    if augment:
        augmented = train_transform(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
    
    img = img.astype(np.float32) / 255.0
    mask = mask.astype(np.float32) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    
    return img, mask

def gen(img_list, mask_list, augment=False):
    for p, m in zip(img_list, mask_list):
        try:
            yield load_and_preprocess(p, m, augment)
        except Exception as e:
            print(f"⚠ Skipping: {p} - {e}")

train_dataset = tf.data.Dataset.from_generator(
    lambda: gen(train_img, train_mask, augment=True),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
    )
).shuffle(100).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: gen(val_img, val_mask, augment=False),
    output_signature=(
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 1), dtype=tf.float32),
    )
).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

print("✅ Datasets ready. Building model...")

# ============================
# U-Net Model
# ============================
def build_unet(input_shape=(768, 1024, 3)):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder
    c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    c1 = layers.Conv2D(32, 3, padding="same", activation="relu")(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(p1)
    c2 = layers.Conv2D(64, 3, padding="same", activation="relu")(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(128, 3, padding="same", activation="relu")(p2)
    c3 = layers.Conv2D(128, 3, padding="same", activation="relu")(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(256, 3, padding="same", activation="relu")(p3)
    c4 = layers.Conv2D(256, 3, padding="same", activation="relu")(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)
    
    # Bottleneck
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(p4)
    b = layers.Conv2D(512, 3, padding="same", activation="relu")(b)
    b = layers.Dropout(0.5)(b)
    
    # Decoder
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(b)
    u4 = layers.Concatenate()([u4, c4])
    c5 = layers.Conv2D(256, 3, padding="same", activation="relu")(u4)
    c5 = layers.Conv2D(256, 3, padding="same", activation="relu")(c5)
    
    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(c5)
    u3 = layers.Concatenate()([u3, c3])
    c6 = layers.Conv2D(128, 3, padding="same", activation="relu")(u3)
    c6 = layers.Conv2D(128, 3, padding="same", activation="relu")(c6)
    
    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(c6)
    u2 = layers.Concatenate()([u2, c2])
    c7 = layers.Conv2D(64, 3, padding="same", activation="relu")(u2)
    c7 = layers.Conv2D(64, 3, padding="same", activation="relu")(c7)
    
    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(c7)
    u1 = layers.Concatenate()([u1, c1])
    c8 = layers.Conv2D(32, 3, padding="same", activation="relu")(u1)
    c8 = layers.Conv2D(32, 3, padding="same", activation="relu")(c8)
    
    outputs = layers.Conv2D(1, 1, activation="sigmoid", dtype="float32")(c8)
    
    return models.Model(inputs, outputs)

model = build_unet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# ============================
# DICE + BCE LOSS FUNCTIONS
# ============================
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

def combined_loss(y_true, y_pred):
    # This is what worked in your original training!
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=8e-4,      # This was lowered slightly for stability
        clipvalue=0.2,
        clipnorm=1.0              # Extra protection against gradient explosions, also added for stability
    ),
    loss=combined_loss,
    metrics=["accuracy", dice_coefficient],
)

model.summary()

# ============================
# HISTORY FILE
# Save training history to this file at each epoch.
# ============================
HISTORY_FILE = os.path.join(BASE_DIR, "training_history_stable.pkl")

# ============================
# CALLBACKS WITH NaN DETECTION
# This was added to stop training if NaNs are detected in loss or metrics.
# No nans were produced during the final training, but this was a safegaurd implemented in earlier training attempts.
# ============================
class SaveEveryEpoch(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, history_file):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.history_file = history_file
        self.epoch_history = {'loss': [], 'accuracy': [], 'dice_coefficient': [], 
                             'val_loss': [], 'val_accuracy': [], 'val_dice_coefficient': [], 'lr': []}
        
        if os.path.exists(self.history_file):
            with open(self.history_file, "rb") as f:
                self.epoch_history = pickle.load(f)
    
    def on_epoch_end(self, epoch, logs=None):
        # Check for NaN
        if np.isnan(logs.get('loss', 0)) or np.isnan(logs.get('dice_coefficient', 0)):
            print("\n" + "="*60)
            print("❌ NaN DETECTED - STOPPING TRAINING")
            print("="*60)
            print(f"Training became unstable at epoch {epoch+1}")
            print(f"Best model: {os.path.join(self.checkpoint_dir, 'best_model.keras')}")
            print("="*60)
            self.model.stop_training = True
            return
        
        # Save checkpoint at each epoch 
        path = os.path.join(self.checkpoint_dir, f"epoch-{epoch+1:03d}.keras")
        self.model.save(path)
        print(f"💾 Checkpoint saved: epoch-{epoch+1:03d}.keras")
        
        # Save history
        for key in logs.keys():
            if key in self.epoch_history:
                self.epoch_history[key].append(float(logs[key]))
        
        with open(self.history_file, "wb") as f:
            pickle.dump(self.epoch_history, f)
        print(f"📊 History saved: {len(self.epoch_history['loss'])} total epochs")

callbacks = [
    SaveEveryEpoch(CHECKPOINT_DIR, HISTORY_FILE),
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(CHECKPOINT_DIR, "best_model.keras"),
        save_best_only=True,
        monitor="val_dice_coefficient",
        mode="max",
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=7,  # More patient
        min_lr=1e-7,
        verbose=1
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=20,  # Much more patient
        restore_best_weights=False,
        verbose=1
    ),
]

# ============================
# RESUME LOGIC
# The model resumes from the latest checkpoint if available, including the training history.
# This was added to facilitate the long training time for the full 25 epochs.
# ~7 hours total for the final model on my GPU.
# ============================
initial_epoch = 0

checkpoints = sorted(
    [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("epoch-") and f.endswith(".keras")]
)

if checkpoints:
    latest_checkpoint = checkpoints[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest_checkpoint)
    initial_epoch = int(latest_checkpoint.split("-")[1].split(".")[0])
    
    print(f"🔄 Resuming from {latest_checkpoint} (starting at epoch {initial_epoch})")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={
            'combined_loss': combined_loss,
            'dice_coefficient': dice_coefficient
        }
    )
    
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "rb") as f:
            previous_history = pickle.load(f)
        print(f"📊 Loaded history ({len(previous_history.get('loss', []))} epochs)")
else:
    print("🚀 Starting fresh training.")

# ============================
# TRAIN
# ============================
print("\n" + "="*60)
print("TRAINING STARTING")
print(f"Target: Val Dice > 0.85")
print("="*60 + "\n")

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    callbacks=callbacks,
    verbose=1
)

# ============================
# SAVE FINAL
# Final model save after 25 epochs.
# ============================
model.save(os.path.join(BASE_DIR, "final_model_stable.keras"))
print(f"✅ Training complete!")
