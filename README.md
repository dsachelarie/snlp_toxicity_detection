# snlp_toxicity_detection
Group project by Henry Kivijärvi, Jade Reinilä, and David Sachelarie for the *SNLP Course Competition 2024*, organized by the course *Statistical Natural Language Processing* from **Aalto University**.

Contest description: *The main task of this competition is to develop text-based models capable of classifying short texts as toxic/non-toxic based on their content.*

We assessed the performance of three types of models, as presented in the PDF report:
- **baseline model (SVM)**: BasicModel()
- **CNN model (AlexNet)**: NNModel()
- **fine-tuned transformers**: not included in the repository

Provide *preprocessing="glove"* or *preprocessing="glove_rtf_igm"* (default) to BaselineModel() and NNModel() to choose the type of preprocessing.
Provide *mode="debug"* (default) for getting performance metrics on a labeled development set, and *mode="release"* for getting predictions on an unlabeled test set.
