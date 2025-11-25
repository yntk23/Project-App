all_sequences = []
for store_data in sequences.values():
    for prod_data in store_data.values():
        all_sequences.append(prod_data['sequence'])

X = np.array(all_sequences)  # shape: (N, 7)
y = X[:, -3:].mean(axis=1, keepdims=True)  # shape: (N, 1)