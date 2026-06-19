import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

# ── Paths ──────────────────────────────────────────────
PROCESSED_DIR  = Path("../data/processed")
TRAIN_DIR      = PROCESSED_DIR / "train"
VAL_DIR        = PROCESSED_DIR / "val"
MODEL_SAVE_DIR = Path("../model/saved")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────
IMG_SIZE    = (300, 300)   # EfficientNetB3 native input size
BATCH_SIZE  = 32
SEED        = 42
NUM_CLASSES = 38

# ── Data Loaders ───────────────────────────────────────
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
    label_mode="categorical"   # one-hot encoded labels for 38 classes
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    label_mode="categorical"
)

# ── Save class names ───────────────────────────────────
class_names = train_ds.class_names
print(f"Classes found : {len(class_names)}")
print(f"Train batches : {len(train_ds)}")
print(f"Val batches   : {len(val_ds)}")

# ── Augmentation (train only) ───────────────────────────
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomBrightness(0.2),
], name="augmentation")

# ── Preprocessing function ──────────────────────────────
preprocess = tf.keras.applications.efficientnet.preprocess_input

# ── Apply to datasets ───────────────────────────────────
train_ds = train_ds.map(
    lambda x, y: (preprocess(augmentation(x, training=True)), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    lambda x, y: (preprocess(x), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# ── Performance optimization ────────────────────────────
# prefetch loads next batch while GPU processes current batch
train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("Preprocessing and augmentation applied.")
print("Train and val datasets ready.")

# ── Backbone ────────────────────────────────────────────
backbone = EfficientNetB3(
    include_top=False,          # remove EfficientNet's original classifier
    weights="imagenet",         # use pretrained ImageNet weights
    input_shape=(300, 300, 3)
)
backbone.trainable = False      # freeze all backbone layers

# ── Custom Classification Head ──────────────────────────
inputs  = tf.keras.Input(shape=(300, 300, 3))
x       = backbone(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.BatchNormalization()(x)
x       = layers.Dense(256, activation="relu")(x)
x       = layers.Dropout(0.4)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs, name="CropDiseaseClassifier")

# ── Summary ─────────────────────────────────────────────
model.summary()

# Count trainable vs frozen parameters
trainable     = sum([tf.size(w).numpy() for w in model.trainable_weights])
non_trainable = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
print(f"\nTrainable parameters     : {trainable:,}")
print(f"Non-trainable parameters : {non_trainable:,}")


from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

# ── Compile ─────────────────────────────────────────────
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# ── Callbacks ───────────────────────────────────────────
callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=str(MODEL_SAVE_DIR / "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

print("Model compiled.")
print("Callbacks ready.")