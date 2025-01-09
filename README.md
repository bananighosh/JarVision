# JarVision


**JarVision** is a Multimodal (Vision) Language Model from scratch in PyTorch designed to answer queries about images using the power of Paligemma. Inspired by the fictional AI assistant "Jarvis," JarVision aims to be a versatile and intelligent image-question-answering system capable of providing insightful responses from visual data.

## Project Objective

JarVision is a vision-language model designed from scratch with the following goals:

* Accept an image input and a natural language query.
* Generate contextually relevant answers using a multimodal architecture.
* Leverage Paligemma for language understanding and generation.

## How It Works

* **Image Input**: The user uploads an image.
* **Text Query**: A natural language query related to the image is provided.
* **Multimodal Processing**: The image and query are processed using a custom vision transformer(Siglip) and Paligemma's language model capabilities.
* **Answer Generation**: JarVision generates a detailed and contextually relevant response based on the provided inputs.

## Technologies Used

* Python for core development
* PyTorch for deep learning framework
* Paligemma for language generation
* Vision Transformer (ViT) for image feature extraction


## Acknowledgments

* Inspired by JARVIS from the Marvel Universe
* Powered by Paligemma and Vision Transformers