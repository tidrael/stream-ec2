from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTFeatureExtractor


class ImageClassifier:
    def __init__(self, max_length=14, num_beams=4):
        self.model = VisionEncoderDecoderModel.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            "nlpconnect/vit-gpt2-image-captioning"
        )
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    def predict(self, image):
        images = []
        images.append(image)
        pixel_values = self.feature_extractor(
            images=images, return_tensors="pt"
        ).pixel_values
        output_ids = self.model.generate(pixel_values, **self.gen_kwargs)
        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds
