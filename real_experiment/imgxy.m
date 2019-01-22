
function focus_change()
% This function is to change the focus point of a image to create a obscure effect.
img = imread('./demo/0000_color.png');
figure(1);
imshow(img);
hold on;
[x, y] = ginput(1)
end