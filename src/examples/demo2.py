import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import radixdb.evaluator
import io

c=radixdb.evaluator.eval("Q().tables(T.animals).fields('*').hist(bins=3)")
ret = io.BytesIO(c['output']['data'])
ret.seek(0)
img=mpimg.imread(ret)
imgplot = plt.imshow(img)
plt.show()
