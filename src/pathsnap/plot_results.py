from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def main(results=Path("results")):
    d=pd.read_csv(results/"model_comparison.csv")
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(d.model,d.cv_macro_f1_mean,yerr=d.cv_macro_f1_std,capsize=4)
    ax.set_ylabel("CV Macro F1");ax.set_title("GroupKFold candidate-classification Macro F1")
    ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(results/"cv_macro_f1_comparison.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    x=range(len(d));w=.36
    ax.bar([i-w/2 for i in x],d.student_top1_accuracy,w,label="Top-1 accuracy")
    ax.bar([i+w/2 for i in x],d.student_macro_f1,w,label="Macro F1")
    ax.set_xticks(list(x),d.model,rotation=20);ax.set_ylim(0,1);ax.set_ylabel("Score")
    ax.set_title("Student-test observation-level performance");ax.legend()
    fig.tight_layout();fig.savefig(results/"student_metrics_comparison.png",dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.bar(d.model,d.student_log_loss)
    ax.set_ylabel("Binary candidate log loss (lower is better)");ax.set_title("Student-test candidate probability quality")
    ax.tick_params(axis="x",rotation=20);fig.tight_layout();fig.savefig(results/"log_loss_comparison.png",dpi=180);plt.close(fig)
    h=pd.read_csv(results/"hmm_comparison.csv")
    # Add route continuity from the saved per-observation predictions.
    preds=pd.read_csv(results/"hmm_predictions.csv")
    # route continuity is computed separately by the training pipeline.
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(h.method,h.top1_accuracy)
    ax.set_ylim(0,1);ax.set_ylabel("Observation-level Top-1 accuracy")
    ax.set_title("Classifier-only vs HMM/Viterbi")
    fig.tight_layout();fig.savefig(results/"hmm_accuracy_comparison.png",dpi=180);plt.close(fig)
    if (results/"tuned_vs_untuned.csv").exists():
        t=pd.read_csv(results/"tuned_vs_untuned.csv")
        fig,ax=plt.subplots(figsize=(8,4))
        xx=range(len(t));ww=.22
        for i,col in enumerate(["cv_macro_f1","student_top1_accuracy","student_macro_f1"]):
            ax.bar([j+(i-1)*ww for j in xx],t[col],ww,label=col)
        ax.set_xticks(list(xx),t.model,rotation=10);ax.set_ylim(0,1);ax.legend()
        ax.set_title("Untuned vs Optuna-tuned Random Forest")
        fig.tight_layout();fig.savefig(results/"tuning_comparison.png",dpi=180);plt.close(fig)

if __name__=="__main__":main()
