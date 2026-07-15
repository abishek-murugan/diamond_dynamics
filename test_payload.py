import pickle

class CumlDummy:
    @classmethod
    def host_deserialize(cls, header, frames):
        return frames[0]

class CumlUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cuml'):
            return CumlDummy
        return super().find_class(module, name)

with open('models/best_clustering_model.pkl', 'rb') as f:
    model = CumlUnpickler(f).load()

if hasattr(model, 'cluster_centers_'):
    print(model.cluster_centers_.__dict__)
