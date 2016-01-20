from keras.utils.visualize_util import plot
from keras.models import model_from_yaml

# load the saved model
model = model_from_yaml(open('imdb_model.yml').read())
model.load_weights('imdb_model_weights.h5')

pred = model.predict('Something', batch_size=128, verbose=0)

print(pred)
