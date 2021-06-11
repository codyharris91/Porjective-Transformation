Projective Transformation (Homography)
======================================

### How to Use
```
python homography.py [city_image] [image_to_put_on_billboard] [output_name]
```

1.  A window will pop-up with the image with blank billboards

2.  Double click the corners of the empty billboard

3.  Type 'Enter' when you have clicked exactly 4 corners. (If you click more or
    less points, the program will terminate)

4.  The billboard image with your new image will be displayed to the screen as
    well as saved to the /output folder.

### Examples
```
python homography.py input_images/part1/input/target/empty_bilboard4.jpg
input_images/part1/input/source/gcat.jpg grumpycat
```

![example1](/examples/grumpycat.png)
```
python homography.py input_images/part1/input/target/empty_bilboard1.jpg
input_images/part1/input/source/happy_minions1.png minion1
```
![example2](/examples/minion1.png) 

```
python homography.py input_images/part1/input/target/empty_bilboard2.jpg
input_images/part1/input/source/happy_minions2.png minion2
```

![example3](/examples/minion2.png)

### Implementation Details

OpenCV was used to annotate the image to have the 4 corners of an empty
billboard marked.

The four points were then ordered so that they would match the following
pattern, where the number shown is the index of the point.

0 \> 1  
4 \< 2

This was neccessary so that the top left corner was always mapped to the top
left corner of the billboard image. The points for the corners of the image
being put on the billboard were just built using the following method using the
width and height of the image:

((0, 0), (bb_width, 0), (bb_width, bb_height), (0, bb_height))

### Putting Pixels on the Board

Instead of grabbing each pixel from the image that is being pasted on the
billboard and translating it to the target image, we take the inverse of the H
matrix and loop through the target image to find what pixels map to the
billboard image. If we do not do this, then there will be white pixels if the
image we are putting on the billboard is smaller than the actual area it is
being applied to.

 
-
