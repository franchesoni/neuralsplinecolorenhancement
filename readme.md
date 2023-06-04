# to-do
- 3d lut oracle
- demo competitors (inference one image given path)
- eval competitors 


# results
simplest with penalty 3 channels 12.08 76 val
simplest with 7 axis 11.33 76 val

- hparams: number nodes, axis
- training axis or a priori
- splines (cubic, simplest, gaussian) 1D
- architecture


# Data
We use MIT 5K with the random250 test split.
We create a validation split with the last 250 images of the sorted trainval split.
The train dataset is preprocessed by resizing inputs and targets to 448x448 and mapping the input to [0,1], this gives the file `processed_train`





