import tensorflow as tf;
import os;
import cv2;
import numpy as np;
import tqdm;
from sklearn.preprocessing import LabelBinarizer;
import matplotlib.pyplot as plt


BASE_PATH = '/Users/nibabi/Desktop/skateboard_trick_classification/Tricks'
VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.mov')
SEQUENCE_LENGTH = 300


# 抽取视频关键帧

def frame_generator():
    video_paths = tf.io.gfile.glob(VIDEOS_PATH)
    np.random.shuffle(video_paths)
    for video_path in video_paths:
        
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
        current_frame = 0
        
        max_images = SEQUENCE_LENGTH
        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_frame % sample_every_frame == 0:
                # OPENCV reads in BGR, tensorflow expects RGB so we invert the order
                frame = frame[:, :, ::-1]
                img = tf.image.resize(frame, (299, 299))
                img = tf.keras.applications.inception_v3.preprocess_input(img)
                max_images -= 1
                
                yield img, video_path

            current_frame += 1

            if max_images == 0:
                break

dataset = tf.data.Dataset.from_generator(frame_generator,
             output_types=(tf.float32, tf.string),
             output_shapes=((299, 299, 3), ()))

dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

# 利用inception_v3进行特征提取
inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

x_f = inception_v3.output

pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x_f)

feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)

current_path = None
all_features = []

for img, batch_paths in tqdm.tqdm(dataset):
    batch_features = feature_extraction_model(img)
    batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1))
    
    for features, path in zip(batch_features.numpy(), batch_paths.numpy()):
        if path != current_path and current_path is not None:
            output_path = current_path.decode().replace('.mov', '.npy')
            np.save(output_path, all_features)
            all_features = []
            
        current_path = path
        all_features.append(features)
        
if all_features:
    output_path = current_path.decode().replace('.mov', '.npy')
    np.save(output_path, all_features)
        
# 模型
LABELS = ['Ollie','Kickflip','Shuvit'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

def generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS, NUM_CELLS=8):

    ip = tf.keras.Input(shape=(MAX_SEQUENCE_LENGTH,2048))

    x = tf.keras.layers.LSTM(NUM_CELLS)(ip)
    x = tf.keras.layers.Dropout(0.5)(x)

    y = tf.keras.layers.Permute((2, 1))(ip)
    y = tf.keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y =  tf.keras.layers.Dropout(0.5)(y)

    y = tf.keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y =  tf.keras.layers.Dropout(0.5)(y)

    y = tf.keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation('relu')(y)
    y =  tf.keras.layers.Dropout(0.5)(y)

    y = tf.keras.layers.GlobalAveragePooling1D()(y)

    x = tf.keras.layers.concatenate([x, y])
    
    x = tf.keras.layers.Dense(512, activation='relu')(x)  # 添加额外的全连接层
    x = tf.keras.layers.Dropout(0.5)(x)

    out = tf.keras.layers.Dense(NB_CLASS, activation='softmax')(x)

    model = tf.keras.Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


with open('/Users/nibabi/Desktop/skateboard_trick_classification/Test_list.txt') as f:
    test_list = [row.strip() for row in list(f)]

with open('/Users/nibabi/Desktop/skateboard_trick_classification/Train_list.txt') as f:
    train_list = [row.strip() for row in list(f)]
    train_list = [row.split(' ')[0] for row in train_list]
    
    
def make_generator(file_list):
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            full_path = os.path.join(BASE_PATH + '/', path).replace('.mov', '.npy')

            label = os.path.basename(os.path.dirname(path))
            features = np.load(full_path)

            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            padded_sequence[0:len(features)] = np.array(features)

            transformed_label = encoder.transform([label])
            yield padded_sequence, transformed_label[0]
    return generator

train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


# 构建模型
MAX_SEQUENCE_LENGTH = 300
NB_CLASS = 3
model = generate_lstmfcn(MAX_SEQUENCE_LENGTH, NB_CLASS)

# 编译模型
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              metrics=['accuracy', 'top_k_categorical_accuracy'])


tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='log', update_freq=1000)
history = model.fit(train_dataset, epochs=50, callbacks=[tensorboard_callback], validation_data=valid_dataset)

# 提取训练和验证的损失和准确度
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# 绘制损失曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制准确度曲线
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()