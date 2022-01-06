# -*-coding: utf-8-*-
import numpy as np
import mmap
import os, sys

class BramConfig(object):
    '''BRAM信息配置'''

    def _construct_block_info(address, size, **offset) -> dict:
        '''构造block信息

            Args:
                name: 块名称
                address: 块起始地址
                size: 块大小
                offset: 偏移量，字典

            Return:
                返回字典，包含address, size, offset字段。
                其中offset是一个字典，表示各块内偏移的用途
        '''
        info = {
            'address': address,
            'size': size,
            'offset': offset
        }
        return info


    block_info = {}
    # 构建新逻辑块参考以下写法
    # 若块内偏移量无特殊含义，则约定key为default，值为0，可根据实际需求修改

    #############################################
    # ir block中包含instr + flag
    # instr包含以下字段：
    #    - uint32 data_startaddr ;  //
    #    - uint32 data_length ;  // byte
    #    - uint32 sync_cycle ;   // 65536
    # flag: uint32
    #############################################

    block_info['data'] = _construct_block_info(
        address=0x40000000, size=8*1024,
        **{'default': 0x0}
    )


class BRAM(object):
    '''实现对Bram读写的类，需要先配置BramConfig类'''
    def __init__(self):
        self.block_info = BramConfig.block_info
        self.block_map = self._mapping('/dev/mem')

    def __del__(self):
        os.close(self.file)
        for block_name, block_map in self.block_map.items():
            block_map.close()

    def _mapping(self, path):
        self.file = os.open(path, os.O_RDWR | os.O_SYNC)

        block_map = {}
        for name, info in self.block_info.items():
            # 构建块内存映射
            block_map[name] = mmap.mmap(
                self.file, 
                info['size'],
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ | mmap.PROT_WRITE,
                offset=info['address']
            )
        return block_map
    
    def write(self, data, block_name: str, offset='default'):
        '''写入数据
            由于数据位宽32bit，因此最好以4的倍数Byte写入(还不知道以1Byte单位写进去会有什么效果)

            输入：
                block_name: BramConfig中配置的block_info的key值
                offset: BramConfig中配置的offset字典key值
        '''
        map_ = self.block_map[block_name]

        # print("Data: \n%s" % data)
        offset_ = self.block_info[block_name]['offset'][offset]
        map_.seek(offset_)

        if isinstance(data, np.ndarray):
            data = data.reshape(-1)
        map_.write(data)

    def read(self, len, block_name, offset='default', dtype=np.uint8) -> np.ndarray:
        '''按字节依次从低字节读取

            输入：
                len: 读取数据长度，单位字节
                block_name: BramConfig中配置的block_info的key值
                offset: BramConfig中配置的offset字典key值
                dtype: 要求数据按相应的格式输出，
                        np.int8, np.int16, np.int32, np.int64,
                        np.uint8, np.uint16, np.uint32, np.uint64
        '''

        # print("Read data from BRAM via AXI_bram_ctrl_1")
        map_ = self.block_map[block_name]

        offset_ = self.block_info[block_name]['offset'][offset]
        map_.seek(offset_)

        # 按字节读取
        # data = []
        # for i in range(len):
        #     data.append(map_.read_byte())
        # data = np.array(data, dtype=np.uint8)
        # data.dtype=dtype   # 按dtype整理数据

        # 按4bytes读取
        data = map_.read(len)
        data = np.frombuffer(data, dtype=dtype)
        return data
