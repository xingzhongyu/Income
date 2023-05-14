import os.path

import numpy as np
import shap
from fastapi import FastAPI, Body, APIRouter
from fastapi.staticfiles import StaticFiles
import time
from matplotlib import pyplot as plt
app=FastAPI()
router = APIRouter()

from getSingleData import Income,get_web_model

income_dataset=None
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/items/file/predict")
async def get_file_predict(net_path:str,data_path:str,file_names:list=Body(embed=True)):#预测接口
    global income_dataset
    if income_dataset is None:
        income_dataset=Income(filenames=file_names,net_path=net_path)
    data=get_web_model(data_path,is_pickle=True)
    pre=income_dataset.predict(data).item()
    return [{"score":pre,"category":pre}]



@app.get("/explanations")
def get_explanations(data_path:str,method:int): #解释接口
    if isinstance(income_dataset,Income):
        data = get_web_model(data_path, is_pickle=True)
        exp=income_dataset.explain(data,method)
        file_path=os.path.join('static',str(time.time()) + "exp.npy")
        np.save(file_path,exp)
        fig = shap.summary_plot(exp, data, plot_type='bar', show=False)
        show_file_path = os.path.join('static',str(time.time()) + "exp.png")
        plt.savefig(show_file_path)
        return {"file_path":file_path,"show_file_path":show_file_path}

@app.get("/evaluate")
async def evaluate(data_path:str,method:int): #解释评估接口
    if isinstance(income_dataset, Income):
        data = get_web_model(data_path, is_pickle=True)
        score=income_dataset.eval_exp(data,method,True)
        return score