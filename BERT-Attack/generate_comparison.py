import model_zoo
import torch
import transformers
import glob
import json

def get_all_files(directory_pattern = "data_defense/outputs/*.tsv"):
    return glob.glob(directory_pattern)

def open_json(filename):
    f = open(filename)
    return json.load(f)

def get_classifer_configs(classifier_name):
    configs = model_zoo.seqClassifiers[classifier_name]
    tokenizer = configs[0].from_pretrained(configs[2])
    model = configs[1].from_pretrained(configs[2])

    return tokenizer, model

def get_model_output(model, tokenizer, input):
    inputs = tokenizer(input, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()


    return model.config.id2label[predicted_class_id]

def run_sequence_classifier(classifier_name, filename):
    original_seq_name = "seq_a"
    adv_seq_name = "adv"

    gen_outputs = open_json(filename)
    tokenzier, model = get_classifer_configs(classifier_name)

    disrupted_count = 0
    nondisrupted_count = 0

    for input_entry in gen_outputs:
        if get_model_output(model, tokenzier, input_entry[original_seq_name]) == get_model_output(model, tokenzier, input_entry[adv_seq_name]):
            nondisrupted_count += 1
        else:
            disrupted_count += 1
    
    accuracy = nondisrupted_count / (disrupted_count + nondisrupted_count)
    return disrupted_count, nondisrupted_count, accuracy


def test_file(json_filename):
    for seq_classifier in model_zoo.seqClassifiers.keys():
        disrutpted, nondisrupted, accuracy = run_sequence_classifier(seq_classifier, json_filename)
        print("Model: %s | Disrupted: %d | Nondisrupted: %d | Accuracy: %f" %(seq_classifier, disrutpted, nondisrupted, accuracy))


def main():
    file_dirs = get_all_files()

    for filename in file_dirs:
        print(filename+":\n")
        test_file(filename)
        print("\n\n")

if __name__ == "__main__":
    main()
