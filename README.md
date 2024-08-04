# LLM_Translate

**LLM_Translate** is an advanced machine learning model designed to bridge the language gap between English and Ukrainian. Leveraging the Tatoeba dataset, this project aims to achieve highly accurate translations through continuous training and improvement. 

## Project Overview

**LLM_Translate** focuses on translating English text to Ukrainian with high precision. Trained over an intensive 24-hour period, the model has achieved notable progress, showing a consistent decrease in training loss across epochs. Here are some key highlights from the training process:

- **Training Duration:** 24 hours
- **Epochs:** 70
- **Training Loss:** Reduced from 1.923 to 0.749

## Training Insights

The model exhibits steady improvement with each epoch, evidenced by the decreasing training loss values. Here are some training milestones:

- **Epoch 60:** Train Loss: 0.783
- **Epoch 65:** Train Loss: 0.765
- **Epoch 70:** Train Loss: 0.749

Example translations generated by the model during training:

1. **Правда , як твої ?** ("True, how are yours?")
2. **Переконайтеся , як ти ?** ("Make sure, how are you?")
3. **Що , як ти ?** ("What, how are you?")
4. **Удачі , як ти ?** ("Good luck, how are you?")
5. **Удачі , Самі ?** ("Good luck, Sam?")

While the results are promising, continuous training is necessary to achieve even better accuracy and fluency.

## Libraries and Frameworks Used

The project is built using the following libraries and frameworks:

- **PyTorch:** For building and training the neural network models.
- **Transformers:** From Hugging Face, used for leveraging pre-trained language models.
- **Tatoeba:** Dataset used for training the translation model.
- **NumPy:** For numerical computations.
- **Pandas:** For data manipulation and preprocessing.
- **Torchtext:** For text processing utilities.
- **Matplotlib:** For plotting training loss and other metrics.

## Model Performance
Despite the promising results, LLM_Translate is still a work in progress. Continuous training and dataset augmentation are essential to further refine the translation quality.

## Contributing
Contributions are welcome! If you have any ideas or improvements, feel free to fork the repository and submit a pull request.
