import numpy as np
import tensorflow as tf
from compound import AtomFeaturizer, BondFeaturizer, graphs_from_smiles
from protein import AAseq_process


class Preprocess:

    def __init__(self, maxlen):
        self.atom_featurizer = AtomFeaturizer(
            allowable_sets={
                "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
                "n_valence": {0, 1, 2, 3, 4, 5, 6},
                "n_hydrogens": {0, 1, 2, 3, 4},
                "hybridization": {"s", "sp", "sp2", "sp3"},
            }
        )
        self.bond_featurizer = BondFeaturizer(
            allowable_sets={
                "bond_type": {"single", "double", "triple", "aromatic"},
                "conjugated": {True, False},
            }
        )
        self.maxlen = maxlen

    
    def prepare_batch(self, x_batch, y_batch):
        """Merges (sub)graphs of batch into a single global (disconnected) graph
        """

        atom_features, bond_features, pair_indices, preprocessed_AAseq = x_batch

        # Obtain number of atoms and bonds for each graph (molecule)
        num_atoms = atom_features.row_lengths()
        num_bonds = bond_features.row_lengths()

        # Obtain partition indices (molecule_indicator), which will be used to
        # gather (sub)graphs from global graph in model later on
        molecule_indices = tf.range(len(num_atoms))
        molecule_indicator = tf.repeat(molecule_indices, num_atoms)

        # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
        # 'pair_indices' (and merging ragged tensors) actualizes the global graph
        gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
        increment = tf.cumsum(num_atoms[:-1])
        increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
        pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
        pair_indices = pair_indices + increment[:, tf.newaxis]
        atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
        bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

        return (atom_features, bond_features, pair_indices, molecule_indicator, preprocessed_AAseq), y_batch


    def MPNN_Transformer_dataset(self, X, y, batch_size=32, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
        if shuffle:
            dataset = dataset.shuffle(1024)
        return dataset.batch(batch_size).map(self.prepare_batch, -1).prefetch(-1)
    

    def get_train_valid_dataset(self, df, compound_column, protein_column, label_column=None, processed_AAseq=None, test_size=0.5):
        if not processed_AAseq:
            self.processed_AAseq = AAseq_process(df[protein_column])
            processed_AAseq = self.processed_AAseq 
        
        # Shuffle array of indices
        permuted_indices = np.random.permutation(np.arange(df.shape[0]))

        # Train set
        train_index = permuted_indices[:int(df.shape[0] * (1-test_size))]
        x_train_compound = graphs_from_smiles(df.iloc[train_index][compound_column], self.atom_featurizer, self.bond_featurizer)
        x_train_protein = processed_AAseq.preprocess_AAseq(df.iloc[train_index][protein_column], self.maxlen)
        if label_column:
            y_train = df.iloc[train_index][label_column]
        else:
            y_train = [None] * len(train_index)
        train_dataset = self.MPNN_Transformer_dataset((x_train_compound[0], x_train_compound[1], x_train_compound[2], x_train_protein), y_train)
        self.train_atom_dim = x_train_compound[0][0][0].shape[0]
        self.train_bond_dim = x_train_compound[1][0][0].shape[0]

        # Valid set
        if test_size:
            valid_index = permuted_indices[int(df.shape[0] * (1-test_size)):]
            x_valid_compound = graphs_from_smiles(df.iloc[valid_index][compound_column], self.atom_featurizer, self.bond_featurizer)
            x_valid_protein = processed_AAseq.preprocess_AAseq(df.iloc[valid_index][protein_column], self.maxlen)
            if label_column:
                y_valid = df.iloc[valid_index][label_column]
            else:
                y_valid = [None] * len(valid_index)
            valid_dataset = self.MPNN_Transformer_dataset((x_valid_compound[0], x_valid_compound[1], x_valid_compound[2], x_valid_protein), y_valid)
            return train_dataset, valid_dataset
        else:
            return train_dataset



