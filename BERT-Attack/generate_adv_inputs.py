import model_zoo
import os

def main(start = 0, end = 250, bpe = 1, sim_mat = 1):
    #template = "--data_path data_defense/imdb_1k.tsv --mlm_path bert-base-uncased --tgt_path models/imdbclassifier --use_sim_mat 0 --output_dir data_defense/imdb_logs_0.tsv --num_label 2 --use_bpe 1 --k 48 --start 0 --end 3 --threshold_pred_score 0 --maskedLM "
    for key in model_zoo.seqClassifiers.keys():
        print(key+":\n")
        command = "--data_path data_defense/imdb_1k.tsv --mlm_path bert-base-uncased --tgt_path models/imdbclassifier --use_sim_mat "+ str(sim_mat) +" --output_dir data_defense/outputs/imdb_logs_"+key+".tsv --num_label 2 --use_bpe "+ str(bpe) +" --k 48 --start "+ str(start) +" --end "+ str(end)+" --threshold_pred_score 0 --maskedLM "+key
        python_path = "python bertattack.py "
        os.system(python_path + command)
        print((15*'*')+"\n\n")

if __name__ == "__main__":
    main()