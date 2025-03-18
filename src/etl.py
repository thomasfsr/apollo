import pickle as pk
import numpy as np
import pandas as pd
import os

class ETL:
    def __init__(self, input_path:str, output_path:str):
        self.input_path = input_path
        self.output_path = output_path

    def extract(self)->dict:
        with open(self.input_path, 'rb') as f:
            return pk.load(f)
        
    def transform(self, data:dict, dim_emb:int)->pd.DataFrame:
        dict = {
            'syndrome_id':[],
            'subject_id':[],
            'image_id':[]
        }
        embeddings = [f'd_{i+1}' for i in range(dim_emb)]
        dict.update({i:[] for i in embeddings})

        emb = []
        for i in data.keys():
            d = data[i]
            for j in d.keys():
                e = d[j]
                for k in e.keys():
                    dict['syndrome_id'].append(i)
                    dict['subject_id'].append(j)
                    dict['image_id'].append(k)
                    emb.append(e[k])
        emb = np.array(emb)
        for i in range(dim_emb):
            dict[embeddings[i]] = emb[:,i]
        df = pd.DataFrame(dict)
        return df
    
    def load(self, df:pd.DataFrame):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        try:
            df.to_csv(self.output_path, index=False)
            print(f"Data saved to {self.output_path}")
        except Exception as e:
            print(f"Error saving data: {e}")

if __name__ == '__main__':
    dim_emb = 320
    etl = ETL('data/mini_gm_public_v0.1.p', 'data/df.csv')
    data = etl.extract()
    df = etl.transform(data, dim_emb)
    etl.load(df)