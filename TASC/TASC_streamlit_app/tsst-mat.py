import scipy.io as sio


def import_mat_file():
    mat_fname = 'C:\\Users\\Public\\tasc-lab\\files\\matfiles\\HA033080917CHR1C02BT54CON0NNN0NNN0NNN0NNN0WH00.mat'
    mat_contents = sio.loadmat(mat_fname)
    # print(sorted(mat_contents.keys()))
    print(mat_contents['At'])


if __name__ == '__main__':
    import_mat_file()
