import matplotlib.pyplot as plt
import pickle
import numpy as np

thresh = np.arange(0.5, 1, 0.05)

r_or_a = "ap"

with open("faster_rcnn_stage_" + r_or_a + ".pkl", 'rb') as fp:
    f_r = pickle.load(fp)
fp.close()

with open("cascade_1st_stage_" + r_or_a + ".pkl", 'rb') as fp:
    c_1_r = pickle.load(fp)
fp.close()

with open("cascade_2nd_stage_" + r_or_a + ".pkl", 'rb') as fp:
    c_2_r = pickle.load(fp)
fp.close()

with open("cascade_3rd_stage_" + r_or_a + ".pkl", 'rb') as fp:
    c_3_r = pickle.load(fp)
fp.close()

plt.title(r_or_a.upper() + " Compare with Faster RCNN & Cascade RCNN 3 stages")
plt.plot(thresh, f_r, 'ro-', label="Faster RCNN " + r_or_a)
plt.plot(thresh, c_1_r, 'bo-', label="Cascade RCNN 1st " + r_or_a)
plt.plot(thresh, c_2_r, 'go-', label="Cascade RCNN 2nd " + r_or_a)
plt.plot(thresh, c_3_r, 'yo-', label="Cascade RCNN 3rd " + r_or_a)

plt.xlabel('Thresh')
plt.ylabel(r_or_a)
plt.ylim([0.0, 1.05])
plt.xlim([0.5, 1.0])

plt.legend()
plt.show()