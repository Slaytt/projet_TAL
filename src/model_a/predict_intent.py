from datasets import load_dataset
import pandas as pd

print("Téléchargement du Switchboard Dialogue Act Corpus en cours...")
# 1. On télécharge le dataset propre depuis le hub
dataset = load_dataset("swda", trust_remote_code=True)

# 2. On convertit directement les données d'entraînement en DataFrame Pandas
df_explo = pd.DataFrame(dataset['train'])

# 3. On filtre pour ne garder que la réplique (text) et l'intention (damsl_act_tag)
df_propre = df_explo[['text', 'damsl_act_tag']].copy()

print("\nVoici tes premières données :")
print(df_propre.head(15))