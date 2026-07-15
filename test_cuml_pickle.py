import pickle
import sklearn.cluster

class CumlDummy:
    @classmethod
    def host_deserialize(cls, header, frames):
        print("Header:", header)
        print("Frames len:", len(frames))
        return sklearn.cluster.KMeans(n_clusters=3)

class CumlUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith('cuml'):
            return CumlDummy
        return super().find_class(module, name)

with open('models/best_clustering_model.pkl', 'rb') as f:
    model = CumlUnpickler(f).load()
print(model)
