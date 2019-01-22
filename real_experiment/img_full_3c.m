function img_out = img_full_3c(img_in)
    left = zeros(56,512,3);
    img_tem = [img_in; left];
    right = zeros(480,128,3);
    img_out = [img_tem, right];
end