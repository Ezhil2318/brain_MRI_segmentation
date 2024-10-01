import tensorflow as tf
from models.unet_pp import unet_pp
from models.attention_unet import attention_unet
from train.loss import dice_loss

# Load data
X_train, X_test, y_train, y_test = ...  # Assume preprocessing done

# Choose a model: U-Net++ or Attention U-Net
model = unet_pp(input_shape=(256, 256, 1))  # Or attention_unet

# Compile the model
model.compile(optimizer='adam', loss=dice_loss, metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=16)

# Save the best model
model.save("../checkpoints/best_model.h5")
