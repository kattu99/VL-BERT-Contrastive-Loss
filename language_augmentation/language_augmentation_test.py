from language_augmentation import LanguageAugmentation

vocab = [("tree", 100), ("flower", 100), ("fire", 100), ("boat", 100)]

question = "how tall is the "

augment_object = LanguageAugmentation(vocab)

print(augment_object.augment_sentence(["how", "tall", "is", "the", "dog"]))
