function lf = read_eslf(read_path, an_org, an_new)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read [h,w,3,ah,aw] data from eslf data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

eslf = im2uint8(imread(read_path));

H = size(eslf,1) / an_org;
H = floor(H/4)*4;
W = size(eslf,2) / an_org;
W = floor(W/4)*4;

lf = zeros(H,W,3,an_org,an_org,'uint8');

for v = 1:an_org
    for u = 1:an_org
        sub = eslf(v:an_org:end, u:an_org:end, :);
        lf(:,:,:,v,u) = sub(1:H,1:W,:);
    end
end
an_crop = ceil((an_org - an_new) / 2 );
lf = lf(:,:,:,1+an_crop:an_new+an_crop,1+an_crop:an_new+an_crop);

end
