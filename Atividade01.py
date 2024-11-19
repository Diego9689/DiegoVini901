import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
import joblib
import os


class Modelo:
    def __init__(self):
        self.df = None
        self.models = None
        self.X_test = None
        self.y_test = None

    def CarregarDataset(self, paths):
        # Suporte para múltiplos datasets
        if isinstance(paths, list):
            datasets = [pd.read_csv(path) for path in paths]
            self.df = pd.concat(datasets, ignore_index=True)
        else:
            self.df = pd.read_csv(paths)

        print(f"Dataset carregado com {len(self.df)} amostras e {len(self.df.columns)} atributos.")

    def TratamentoDeDados(self):
        # Converter rótulos para códigos numéricos
        if 'Species' not in self.df.columns:
            raise ValueError("A coluna 'Species' não foi encontrada no dataset.")

        self.df['Species'] = self.df['Species'].astype('category').cat.codes

        # Verificar valores nulos
        if self.df.isnull().sum().any():
            print("Dados contêm valores nulos. Removendo...")
            self.df.dropna(inplace=True)

        # Balancear dados usando SMOTE
        X = self.df.drop('Species', axis=1)
        y = self.df['Species']
        smote = SMOTE(random_state=42)
        X_bal, y_bal = smote.fit_resample(X, y)

        # Atualizar dataframe
        self.df = pd.concat([pd.DataFrame(X_bal, columns=X.columns), pd.Series(y_bal, name='Species')], axis=1)
        print(f"Dados balanceados. Total: {len(self.df)} amostras.")

    def Treinamento(self):
        # Separar dados em treino e teste
        X = self.df.drop('Species', axis=1)
        y = self.df['Species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inicializar modelos
        svm_model = SVC(kernel='linear', random_state=42)
        lr_model = LogisticRegression(max_iter=200, random_state=42)
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        knn_model = KNeighborsClassifier(n_neighbors=5)

        # Treinar modelos
        svm_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        rf_model.fit(X_train, y_train)
        knn_model.fit(X_train, y_train)

        # Armazenar modelos
        self.models = {
            'SVM': svm_model,
            'Logistic Regression': lr_model,
            'Random Forest': rf_model,
            'KNN': knn_model
        }
        self.X_test = X_test
        self.y_test = y_test
        print("Modelos treinados com sucesso.")

    def Teste(self):
        results = {}
        for model_name, model in self.models.items():
            y_pred = model.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            cm = confusion_matrix(self.y_test, y_pred)
            report = classification_report(self.y_test, y_pred, output_dict=True)

            results[model_name] = {
                'accuracy': acc,
                'confusion_matrix': cm,
                'classification_report': report
            }

            # Exibir resultados
            print(f"\nModelo: {model_name}")
            print(f"Acurácia: {acc:.2f}")
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Setosa', 'Versicolor', 'Virginica'],
                        yticklabels=['Setosa', 'Versicolor', 'Virginica'])
            plt.title(f"Matriz de Confusão - {model_name}")
            plt.xlabel("Predito")
            plt.ylabel("Real")
            plt.show()

        return results

    def SalvarModelos(self, output_dir="modelos"):
        os.makedirs(output_dir, exist_ok=True)
        for model_name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f"{model_name}_modelo.pkl"))
        print(f"Modelos salvos no diretório: {output_dir}")

    def Train(self, dataset_paths):
        self.CarregarDataset(dataset_paths)
        self.TratamentoDeDados()
        self.Treinamento()
        resultados = self.Teste()
        self.SalvarModelos()
        return resultados


if __name__ == "__main__":
    # Caminhos para múltiplos datasets (pode ser um único caminho ou lista de caminhos)
    datasets = [
        "C:/Users/usuario/Downloads/iris.data",
        "C:/Users/usuario/Downloads/iris_extra.data"
    ]

    modelo = Modelo()
    resultados = modelo.Train(datasets)
