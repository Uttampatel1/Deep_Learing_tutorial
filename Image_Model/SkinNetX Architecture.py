## Images through Concatenated MobileNetV2 and Xception Models

## Full Code : https://www.kaggle.com/code/utampipaliyagmailcom/skin-cancer-clf-with-skinnetx-architecture

# Model definition remains the same
# Define the model architecture (MobileNetV2 + Xception as previously defined)
mobile_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
mobile_model.trainable = False  # Freeze MobileNetV2 layers

xception_model = tf.keras.applications.Xception(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
xception_model.trainable = False  # Freeze Xception layers

# Extract features
mobile_output = mobile_model.output
xception_output = xception_model.output
mobile_output = tf.keras.layers.Conv2D(256, (1, 1))(mobile_output)
xception_output = tf.keras.layers.Conv2D(256, (1, 1))(xception_output)

# Concatenate features from both models
concat_features = tf.keras.layers.Concatenate()([mobile_output, xception_output])
conv_layer = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same')(concat_features)
flattened_features = tf.keras.layers.Flatten()(conv_layer)
dropout = tf.keras.layers.Dropout(0.4)(flattened_features)
classifier_output = tf.keras.layers.Dense(2, activation='softmax')(dropout)

# Define the full model with two inputs and one output
model = tf.keras.Model(inputs=[mobile_model.input, xception_model.input], outputs=classifier_output)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
