# to-do
- 3d lut oracle
- demo competitors (inference one image given path)
- eval competitors 


# results
simplest with penalty 3 channels 12.08 76 val
simplest with 7 axis 11.33 76 val (runs/Jun02_17-30-07_weird-power)


- hparams: number nodes, axis
- training axis or a priori
- splines (cubic, simplest, gaussian) 1D
- architecture

====================
val clutnet
--------------------
de76 11.310185
de94 95.77119
mae 0.06297801
psnr 23.209019
mse 0.0074469736
ssim 0.84765625
====================
val curl
--------------------
de76 11.007149
de94 102.543205
mae 0.06427708
psnr 22.879978
mse 0.007628095
ssim 0.84547865
====================
val ia3dlut
--------------------
de76 11.2225275
de94 101.27734
mae 0.0648031
psnr 23.0638
mse 0.0077532367
ssim 0.84217143
====================
val maxim
--------------------
de76 9.745365
de94 88.437935
mae 0.05994275
psnr 23.739414
mse 0.0064541986
ssim 0.8506145
====================
val nse
--------------------
de76 14.107348
de94 203.34247
mae 0.10562189
psnr 19.259506
mse 0.017525028
ssim 0.8241527
====================
val oracle3dlut
--------------------
de76 4.298839
de94 19.901468
mae 0.021400109
psnr 31.4657
mse 0.0011626269
ssim 0.90834457
====================
test clutnet
--------------------
de76 11.319051
de94 108.385414
mae 0.06650369
psnr 22.856966
mse 0.008505996
ssim 0.8229735
====================
test curl
--------------------
de76 10.689641
de94 97.94769
mae 0.0628198
psnr 22.886576
mse 0.0074530705
ssim 0.8323023
====================
test ia3dlut
--------------------
de76 11.61062
de94 115.979836
mae 0.07015192
psnr 22.434275
mse 0.009106748
ssim 0.8125029
====================
test maxim
--------------------
de76 10.15736
de94 93.484375
mae 0.06278963
psnr 23.294592
mse 0.006975092
ssim 0.8241023
====================
test nse
--------------------
de76 15.490198
de94 236.42274
mae 0.11587683
psnr 18.29282
mse 0.020981716
ssim 0.7902124
====================
test oracle3dlut
--------------------
de76 4.4504776
de94 20.352753
mae 0.020608082
psnr 31.359543
mse 0.0011194482
ssim 0.8981874


# Data
We use MIT 5K with the random250 test split.
We create a validation split with the last 250 images of the sorted trainval split.
The train dataset is preprocessed by resizing inputs and targets to 448x448 and mapping the input to [0,1], this gives the file `processed_train`





