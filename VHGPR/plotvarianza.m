function plotvarianza(Xs, media, varianza)
gris = [media+2*sqrt(varianza);flipdim(media-2*sqrt(varianza),1)];
fill([Xs; flipdim(Xs,1)], gris, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);