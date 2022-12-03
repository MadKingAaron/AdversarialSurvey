# PIRE Tests

## Generate Images
To generate adversarial images follow the instructions in `./PIRE/PIRE_Evaluiation/readme.md`

## Evaluate
1. To evaluate images copy the images in `./PIRE/PIRE_Evaluation/PIRE/img_input` to `./PIRE/PIRE_Evaluation/gt_queries` and images in `./PIRE/PIRE_Evaluation/PIRE/img_output` to `./PIRE/Image_Comparison/adv_queries`
2. Run `./PIRE/Image_Comparison/get_predictions.py` to get the success rates for each model
