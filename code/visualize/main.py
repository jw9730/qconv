import numpy as np
from matplotlib import pyplot as plt

def main():
    # problem 1
    p1_ctime = [[0.191156, 0.192221, 0.185453, 0.190423, 0.187872], # sample1
                [0.178498, 0.196870, 0.182962, 0.187924, 0.189490], # sample2
                [0.115871, 0.128440, 0.124409, 0.134413, 0.154154]] # sample3
    p1_ctime_avg = np.asarray(p1_ctime).mean()
    print("P1: convolution elapsed time {}s".format(p1_ctime_avg))



    # problem 2
    p2_nrmse_int32 = [[0.00000004084621352263, 0.00000004084621352263, 0.00000004084621352263], # sample1
                   [0.00000006036117383701, 0.00000006036117383701, 0.00000006036117383701], # sample2
                   [0.00000004598248182219, 0.00000004598248182219, 0.00000004598248182219]] # sample3
    p2_nrmse_int16 = [[0.00007786053174640983, 0.00007786053174640983, 0.00007786053174640983],
                   [0.00008945982699515298, 0.00008945982699515298, 0.00008945982699515298],
                   [0.00013071547436993569, 0.00013071547436993569, 0.00013071547436993569]]
    p2_nrmse_int8 =  [[0.00957141164690256119, 0.00957141164690256119, 0.00957141164690256119],
                   [0.01125544402748346329, 0.01125544402748346329, 0.01125544402748346329],
                   [0.01572080887854099274, 0.01572080887854099274, 0.01572080887854099274]]

    p2_ctime_int32 = [[0.142178, 0.155827, 0.114497],
                   [0.143572, 0.146569, 0.114209],
                   [0.149807, 0.147819, 0.115724]]
    p2_ctime_int16 = [[0.132731, 0.147209, 0.116294],
                   [0.136079, 0.140889, 0.115135],
                   [0.138272, 0.148140, 0.112894]]
    p2_ctime_int8 =  [[0.145235, 0.128348, 0.114113],
                   [0.130233, 0.124860, 0.113161],
                   [0.135349, 0.124791, 0.114010]]

    p2_qtime_int32 = [[0.002876, 0.002100, 0.005004],
                   [0.002692, 0.001946, 0.004775],
                   [0.002792, 0.002113, 0.004963]]
    p2_qtime_int16 = [[0.002489, 0.001973, 0.004237],
                   [0.001679, 0.001784, 0.001855],
                   [0.002587, 0.001882, 0.003900]]
    p2_qtime_int8 =  [[0.002679, 0.001734, 0.003533],
                   [0.002292, 0.001590, 0.003671],
                   [0.002551, 0.001549, 0.003657]]

    p2_nrmse_int32_avg = np.asarray(p2_nrmse_int32).mean()
    p2_ctime_int32_avg = np.asarray(p2_ctime_int32).mean()
    p2_qtime_int32_avg = np.asarray(p2_qtime_int32).mean()
    print("P2: [INT32] NRMSE {}, convolution elapsed time {}s, quantization overhead {}".format(p2_nrmse_int32_avg, p2_ctime_int32_avg, p2_qtime_int32_avg))
    p2_nrmse_int16_avg = np.asarray(p2_nrmse_int16).mean()
    p2_ctime_int16_avg = np.asarray(p2_ctime_int16).mean()
    p2_qtime_int16_avg = np.asarray(p2_qtime_int16).mean()
    print("P2: [INT16] NRMSE {}, convolution elapsed time {}s, quantization overhead {}".format(p2_nrmse_int16_avg, p2_ctime_int16_avg, p2_qtime_int16_avg))
    p2_nrmse_int8_avg = np.asarray(p2_nrmse_int8).mean()
    p2_ctime_int8_avg = np.asarray(p2_ctime_int8).mean()
    p2_qtime_int8_avg = np.asarray(p2_qtime_int8).mean()
    print("P2: [INT8] NRMSE {}, convolution elapsed time {}s, quantization overhead {}".format(p2_nrmse_int8_avg, p2_ctime_int8_avg, p2_qtime_int8_avg))



    # problem 3
    p3_nrmse_fp32 =  [[0.00000004068699510640, 0.00000004068699510640, 0.00000004068699510640],
                   [0.00000006167062593931, 0.00000006167062593931, 0.00000006167062593931],
                   [0.00000004176184020821, 0.00000004176184020821, 0.00000004176184020821]]
    p3_nrmse_int32 = [[0.00000004084621352263, 0.00000004084621352263, 0.00000004084621352263],
                   [0.00000006036117383701, 0.00000006036117383701, 0.00000006036117383701],
                   [0.00000004598248182219, 0.00000004598248182219, 0.00000004598248182219]]
    p3_nrmse_int16 =  [[0.00007786053174640983, 0.00007786053174640983, 0.00007786053174640983],
                   [0.00008945982699515298, 0.00008945982699515298, 0.00008945982699515298],
                   [0.00013071547436993569, 0.00013071547436993569, 0.00013071547436993569]]

    p3_ctime_fp32 =  [[0.029275, 0.032564, 0.011134],
                   [0.018063, 0.014191, 0.004449],
                   [0.034504, 0.018394, 0.005916]]
    p3_ctime_int32 = [[0.020713, 0.028409, 0.017898],
                   [0.024839, 0.023282, 0.007078],
                   [0.024557, 0.028438, 0.006474]]
    p3_ctime_int16 = [[0.019049, 0.017318, 0.007351],
                   [0.017566, 0.013928, 0.009028],
                   [0.015899, 0.014058, 0.003802]]

    p3_nrmse_fp32_avg = np.asarray(p3_nrmse_fp32).mean()
    p3_ctime_fp32_avg = np.asarray(p3_ctime_fp32).mean()
    print("P3: [FP32] NRMSE {}, convolution elapsed time {}s".format(p3_nrmse_fp32_avg, p3_ctime_fp32_avg))
    p3_nrmse_int32_avg = np.asarray(p3_nrmse_int32).mean()
    p3_ctime_int32_avg = np.asarray(p3_ctime_int32).mean()
    print("P3: [INT32] NRMSE {}, convolution elapsed time {}s".format(p3_nrmse_int32_avg, p3_ctime_int32_avg))
    p3_nrmse_int16_avg = np.asarray(p3_nrmse_int16).mean()
    p3_ctime_int16_avg = np.asarray(p3_ctime_int16).mean()
    print("P3: [INT16] NRMSE {}, convolution elapsed time {}s".format(p3_nrmse_int16_avg, p3_ctime_int16_avg))



    # problem 4
    p4_nrmse = [[0.00000004988923762994, 0.00000004988923762994, 0.00000004988923762994],
             [0.00000007154167036560, 0.00000007154167036560, 0.00000007154167036560],
             [0.00000001690419892952, 0.00000001690419892952, 0.00000001690419892952]]
    p4_ctime = [[0.005206, 0.005359, 0.005248],
             [0.005006, 0.004712, 0.005139],
             [0.002368, 0.002356, 0.002516]]

    p4_nrmse_avg = np.asarray(p4_nrmse).mean()
    p4_ctime_avg = np.asarray(p4_ctime).mean()
    print("P4: NRMSE {}, convolution elapsed time {}s".format(p4_nrmse_avg, p4_ctime_avg))



    # determining scaling factor
    scaled_nrmse_int32 = [[0.00945814419537782669, 0.01124609448015689850, 0.01541813369840383530], # N=0
                   [0.00485927565023303032, 0.00564367603510618210, 0.00793413445353507996],
                   [0.00244362954981625080, 0.00284413481131196022, 0.00400492642074823380],
                   [0.00119723111856728792, 0.00142925954423844814, 0.00201935949735343456],
                   [0.00061367271700873971, 0.00072000420186668634, 0.00102200929541140795],
                   [0.00030769407749176025, 0.00036024951259605587, 0.00051057414384558797],
                   [0.00015280113439075649, 0.00018034213280770928, 0.00025613314937800169],
                   [0.00007786053174640983, 0.00008945982699515298, 0.00012869243801105767],
                   [0.00003936859138775617, 0.00004456194437807426, 0.00006380445847753435],
                   [0.00001974275619431864, 0.00002243215931230225, 0.00003184317756677046],
                   [0.00000974647900875425, 0.00001127585710491985, 0.00001603061355126556],
                   [0.00000485167493025074, 0.00000564872652830672, 0.00000801567966846051],
                   [0.00000240088525060855, 0.00000284098109659681, 0.00000399683267460205],
                   [0.00000119650360375090, 0.00000139810310884059, 0.00000200275781025994],
                   [0.00000059372359828558, 0.00000069675792246926, 0.00000099913836493215],
                   [0.00000029928261824352, 0.00000035420987387624, 0.00000049667886514726],
                   [0.00000015090333249645, 0.00000017865077950319, 0.00000024530726250305],
                   [0.00000008016011321388, 0.00000009873942730110, 0.00000012352761302736],
                   [0.00000005007403913737, 0.00000006861164791871, 0.00000006748057046480],
                   [0.00000004084621352263, 0.00000006036117383701, 0.00000004598248182219],
                   [0.00000003881108057158, 0.00000005884039566695, 0.00000004040538925665], # N=19
                   [0.00000003846918872341, 0.00000005869118524515, 0.00000003943262072426],
                   [0.00000003841178752850, 0.00000005867803309911, 0.00000003930405156893],
                   [0.00000003840313311798, 0.00000005867443420016, 0.00002019078237935901],
                   [0.00085061544086784124, 0.00008978928963188082, 0.00291984295472502708],
                   [0.04024402424693107605, 0.11319822818040847778, 0.00529492460191249847],
                   [0.09211672097444534302, 0.11072359234094619751, 0.07133841514587402344]]
    scaled_nrmse_int16 = [[0.00945814419537782669, 0.01124609448015689850, 0.01541813369840383530], # N=0
                   [0.00485927565023303032, 0.00564367603510618210, 0.00793413445353507996],
                   [0.00244362954981625080, 0.00284413481131196022, 0.00400492642074823380],
                   [0.00119723111856728792, 0.00142925954423844814, 0.00201935949735343456],
                   [0.00061367271700873971, 0.00072000420186668634, 0.00102200929541140795],
                   [0.00030769407749176025, 0.00036024951259605587, 0.00051057414384558797],
                   [0.00015280113439075649, 0.00018034213280770928, 0.00025613314937800169],
                   [0.00007786053174640983, 0.00008945982699515298, 0.00013071547436993569], # N=7
                   [0.00085539458086714149, 0.00010445084626553580, 0.00292103923857212067],
                   [0.04022435471415519714, 0.11317345499992370605, 0.00529462797567248344]]
    scaled_nrmse_int8  = [[0.00957141164690256119, 0.01125544402748346329, 0.01572080887854099274], # N=0
                   [0.03780232742428779602, 0.10876198858022689819, 0.00949972216039896011]]

    plt.figure()
    plt.plot(2**np.arange(0, len(scaled_nrmse_int32)), np.asarray(scaled_nrmse_int32).mean(axis=1), '-ro', label="INT32")
    plt.plot(2**np.arange(0, len(scaled_nrmse_int16)), np.asarray(scaled_nrmse_int16).mean(axis=1), '-go', label="INT16")
    plt.plot(2**np.arange(0, len(scaled_nrmse_int8)), np.asarray(scaled_nrmse_int8).mean(axis=1), '-bo', label="INT8")
    plt.xlabel("Input scale factor")
    plt.ylabel("NRMSE")
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.plot(2**19, np.asarray(scaled_nrmse_int32).mean(axis=1)[19], 'r+', markersize=25)
    plt.plot(2**7, np.asarray(scaled_nrmse_int16).mean(axis=1)[7], 'g+', markersize=25)
    plt.plot(2**0, np.asarray(scaled_nrmse_int8).mean(axis=1)[0], 'b+', markersize=25)
    plt.savefig("./NRMSE_inscale.png")
    plt.close()



    # performance-accuracy tradeoff
    plt.figure(figsize=(10, 4))

    # including cuda
    plt.subplot(131)
    x = [1, p1_ctime_avg/p2_ctime_int32_avg, p1_ctime_avg/p2_ctime_int16_avg, p1_ctime_avg/p2_ctime_int8_avg]
    y = [0, p2_nrmse_int32_avg, p2_nrmse_int16_avg, p2_nrmse_int8_avg]
    plt.plot(x, y, '-ro', label="Scalar")
    x = [p1_ctime_avg/p3_ctime_fp32_avg, p1_ctime_avg/p3_ctime_int32_avg, p1_ctime_avg/p3_ctime_int16_avg]
    y = [p3_nrmse_fp32_avg, p3_nrmse_int32_avg, p3_nrmse_int16_avg]
    plt.plot(x, y, '-go', label="AVX")
    plt.plot([p1_ctime_avg/p4_ctime_avg], [p4_nrmse_avg], '-bo', label="CUDA (FP32)")
    plt.xlabel("Speedup (times)")
    plt.ylabel("NRMSE")
    plt.legend()

    # quantized scalar
    plt.subplot(132)
    x = [1, p1_ctime_avg/p2_ctime_int32_avg, p1_ctime_avg/p2_ctime_int16_avg, p1_ctime_avg/p2_ctime_int8_avg]
    y = [0, p2_nrmse_int32_avg, p2_nrmse_int16_avg, p2_nrmse_int8_avg]
    plt.plot(x, y, '-ro', label="Scalar")
    plt.annotate('FP32',  xy=[1, 0], textcoords='data', clip_on=True)
    plt.annotate('INT32', xy=[p1_ctime_avg/p2_ctime_int32_avg, p2_nrmse_int32_avg-0.0005], textcoords='data', clip_on=True)
    plt.annotate('INT16', xy=[p1_ctime_avg/p2_ctime_int16_avg, p2_nrmse_int16_avg+0.0002], textcoords='data', clip_on=True)
    plt.annotate('INT8',  xy=[p1_ctime_avg/p2_ctime_int8_avg, p2_nrmse_int8_avg], textcoords='data', clip_on=True)
    plt.xlim([0.95, 1.45])
    plt.xlabel("Speedup (times)")
    plt.ylabel("NRMSE")
    plt.legend()

    # quantized avx
    plt.subplot(133)
    x = [p1_ctime_avg/p3_ctime_fp32_avg, p1_ctime_avg/p3_ctime_int32_avg, p1_ctime_avg/p3_ctime_int16_avg]
    y = [p3_nrmse_fp32_avg, p3_nrmse_int32_avg, p3_nrmse_int16_avg]
    plt.plot(x, y, '-go', label="AVX")
    plt.annotate('FP32',  xy=[p1_ctime_avg/p3_ctime_fp32_avg, p3_nrmse_fp32_avg-0.000003], textcoords='data', clip_on=True)
    plt.annotate('INT32', xy=[p1_ctime_avg/p3_ctime_int32_avg, p3_nrmse_int32_avg+0.000001], textcoords='data', clip_on=True)
    plt.annotate('INT16', xy=[p1_ctime_avg/p3_ctime_int16_avg, p3_nrmse_int16_avg], textcoords='data', clip_on=True)
    plt.xlim([8, 15])
    plt.xlabel("Speedup (times)")
    plt.ylabel("NRMSE")
    plt.legend()

    plt.subplots_adjust(wspace=0.5)
    plt.savefig("./tradeoff.png")
    plt.close()

    # sparsity
    sparsity_int32 = [[22080/222784, 0/36864, 0/200704],
                     [22656/123008, 0/147456, 0/100352],
                     [7680/32768, 1/1048576, 0/100352]]
    sparsity_int16 = [[22287/222784, 15/36864, 53/200704],
                     [22740/123008, 120/147456, 0/100352],
                     [7710/32768, 842/1048576, 0/100352]]
    sparsity_int8  = [[50923/222784, 2495/36864, 53/200704],
                     [34175/123008, 13037/147456, 13/100352],
                     [12433/32768, 101475/1048576, 66/100352]]
    plt.figure(figsize=(4,3))
    sparsity_input = [np.asarray(sparsity_int32).mean(axis=0)[0], np.asarray(sparsity_int16).mean(axis=0)[0], np.asarray(sparsity_int8).mean(axis=0)[0]]
    sparsity_kernel = [np.asarray(sparsity_int32).mean(axis=0)[1], np.asarray(sparsity_int16).mean(axis=0)[1], np.asarray(sparsity_int8).mean(axis=0)[1]]
    sparsity_output = [np.asarray(sparsity_int32).mean(axis=0)[2], np.asarray(sparsity_int16).mean(axis=0)[2], np.asarray(sparsity_int8).mean(axis=0)[2]]
    plt.plot([1, 2, 3], sparsity_input, '-ro', label="Input")
    plt.plot([1, 2, 3], sparsity_kernel, '-go', label="Kernel")
    plt.plot([1, 2, 3], sparsity_output, '-bo', label="Output")
    plt.xticks([1, 2, 3], ['INT32', 'INT16', 'INT8'])
    plt.ylabel('Sparsity')
    plt.ylim([-0.1, 1.1])
    plt.legend()
    plt.savefig("./sparsity.png")
    plt.close()

    # tradeoff
    sparsity_tradeoff = [50923/222784, 2495/36864, 54/200704, 0.00485927565023303032]
    print(np.asarray(sparsity_tradeoff[0:3]))

if __name__ == "__main__":
    main()