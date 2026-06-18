import imageio


def images_to_gif(
    filepath: str, images: list, time_slot: float = 1.0, speed: float = 1.0
):
    """将图像序列转换为 gif

    参数:
      filepath (str): gif 保存文件路径
      images (list): 需要处理的图像列表
      time_slot (float, optional): 相邻帧时间间隔（单位: 秒）。默认为 1.0。
      speed (float, optional): 播放速度倍率，>1 加速，<1 减速。默认为 1.0。
    """
    if speed <= 0:
        raise ValueError("speed 必须大于 0")
    if time_slot <= 0:
        raise ValueError("time_slot 必须大于 0")

    imageio.mimsave(filepath, images, format="gif", duration=time_slot / speed, loop=0)
