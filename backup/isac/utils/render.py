from PIL import Image, ImageDraw, ImageFont, ImageFile
import imageio


def add_text_to_image(
    image: ImageFile,
    text: str,
    position=(10, 10),
    font_path: str = "/home/lhp/fonts/arial.ttf",
    font_size: float = 20,
    color=(0, 0, 0),
):
    """向 `ImageFile` 对象添加文字

    参数:
      image (`ImageFile`): ImageFile 对象。
      text (str): 要添加的文字。
      position (tuple, optional): 文字位置。默认为 (10, 10)。
      font_path (str, optional): 字体文件路径。
      font_size (float, optional): 字体像素大小。默认为 30。
      color (tuple, optional): 文字颜色。默认为 (0, 0, 0) 黑色。

    返回:
      image (`ImageFile`): ImageFile 对象

    异常:
      IOError: 如果字体文件未找到
    """
    # 打开图片
    draw = ImageDraw.Draw(image)
    # 加载字体
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        raise IOError("Font file does't exist!")
    # 添加文字到图片
    draw.text(position, text, fill=color, font=font)
    del draw  # 释放内存

    return image


def draw_line_to_image(
    image_path: str,
    start_point,
    end_point,
    width: int = 5,
    color="red",
    new_image_path: str = None,
):
    """在图片上绘制线段

    参数:
      image_path (str): 原始图片的文件路径。
      start_point (Any): 线段起点。
      end_point (Any): 线段终点。
      width (int, optional): 线宽。默认为 5。
      color (str, optional): 线颜色。默认为 'red'。
      new_image_path (str, optional): 新图片的保存路径。默认为 None。
    """
    image = Image.open(image_path)

    draw = ImageDraw.Draw(image)
    draw.line([start_point, end_point], fill=color, width=width)
    draw.polygon(
        [
            end_point,
            (end_point[0] - 10, end_point[1] - 20),
            (end_point[0] + 10, end_point[1] + 20),
        ],
        fill=color,
    )
    if new_image_path is not None:
        image.save(new_image_path)
    else:
        image.save(image_path)
    del draw  # 释放内存


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
