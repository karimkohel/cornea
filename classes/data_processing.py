
#this is pseudo code just thinking kda to scape form my inside 
#having file contain [(frame,array_of_points,mospos),(frame,array_of_points,mospos)]
#we need to create two approaches 1- make generators 2- use tf pipelining on each batch 
import tensorflow as tf
class CustomDataGen(tf.keras.utils.Sequence):
     def __init__(self, df, X_col(frames,array_of_points), y_col(mospos),
                 batch_size,
                 input_size=(224, 224, 3)
                 ):
          
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size


        self.n = len(self.df)
        self.n_name = df[y_col['name']].nunique()
        self.n_type = df[y_col['type']].nunique()        #msh fahm awi hoa 3ayz ywsl l2eh bs okay 

    #function to generate one batch of data 
    def __get_input(self, path, bbox, target_size): 
    #here we will make the pre processing on the data 

    xmin, ymin, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']

    image = tf.keras.preprocessing.image.load_img(path)
    image_arr = tf.keras.preprocessing.image.img_to_array(image)

    image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
    image_arr = tf.image.resize(image_arr,(target_size[0], target_size[1])).numpy()

    return image_arr/255.    



    def __get_data(self, batches):
        # Generates data containing batch_size samples

        path_batch = batches[self.X_col['path']] #dh al mfrod mkan al sora
        bbox_batch = batches[self.X_col['bbox']] # al sora al mfrod kman wal array_data
        
        name_batch = batches[self.y_col['name']] #mospos bs
        type_batch = batches[self.y_col['type']]

        X_batch = np.asarray([self.__get_input(x, y, self.input_size) for x, y in zip(path_batch, bbox_batch)])

        y0_batch = np.asarray([self.__get_output(y, self.n_name) for y in name_batch])
        y1_batch = np.asarray([self.__get_output(y, self.n_type) for y in type_batch])

        return X_batch, tuple([y0_batch, y1_batch])
    
    def __getitem__(self, index):
        
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size 
    
    traingen = CustomDataGen(train_df,
                         X_col={'path':'filename', 'bbox': 'region_shape_attributes'},
                         y_col={'name': 'name', 'type': 'type'},
                         batch_size=batch_siz, input_size=target_size)
                                                                                            #model 
    valgen = CustomDataGen(val_df,
                       X_col={'path':'filename', 'bbox': 'region_shape_attributes'},
                       y_col={'name': 'name', 'type': 'type'},
                       batch_size=batch_siz, input_size=target_size)

    model.fit(traingen,
          validation_data=valgen,
          epochs=num_epochs)