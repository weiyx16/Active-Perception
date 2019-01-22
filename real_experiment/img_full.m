function img_out = img_full(img_in)
    left = zeros(56,512);
    img_tem = [img_in; left];
    right = zeros(480,128);
    img_out = [img_tem, right];
end