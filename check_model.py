import pickle

with open('models/rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

print('=== INFO MODEL ===')
print(f'Tipe         : {type(model)}')
print(f'Jumlah pohon : {model.n_estimators}')
print(f'Max depth    : {model.max_depth}')
print(f'Fitur input  : {model.n_features_in_}')
print(f'Kelas        : {model.classes_}')
print(f'Feature names: {getattr(model, "feature_names_in_", "(tidak ada)")}')
print(f'OOB Score    : {getattr(model, "oob_score_", "(tidak dipakai)")}')
