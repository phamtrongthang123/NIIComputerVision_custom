import scipy.io
# mat = scipy.io.loadmat('data/FixedPose.mat')
# print("FixedPose")
# print(mat.keys())
# for k,v in mat.items():
#     if k not in ['__header__', '__version__', '__globals__']:
#         print(k, v[0].shape)
#         print(v)
mat = scipy.io.loadmat('data/String4b.mat')
print("String4b")
print(mat.keys())
for k,v in mat.items():
    if k not in ['__header__', '__version__', '__globals__']:
        try:
            print(k, v[0][0][0][0][0][0][0][0].shape)
        except:
            pass
# mat = scipy.io.loadmat('data/Test2Box.mat')
# print("Test2Box")
# print(mat.keys())
# for k,v in mat.items():
#     if k not in ['__header__', '__version__', '__globals__']:
#         print(k, v.shape)
#         print(v)

mat = scipy.io.loadmat('data/SkeletonConnectionMap.mat')
print("SkeletonConnectionMap")
print(mat.keys())
for k,v in mat.items():
    if k not in ['__header__', '__version__', '__globals__']:
        print(k, v.shape)
        print(v)