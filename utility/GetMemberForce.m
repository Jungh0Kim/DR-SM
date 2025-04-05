% Jungho Kim, Get member forces

function [F, u] = GetMemberForce( P,K,T,MN,kl,Fixdof )

Nmember = length(kl);
Nnode = length(P)/2;

P(Fixdof) = [];
K_ = K;
K_(:,Fixdof) = []; K_(Fixdof,:) = [];
u_ = K_\P;
u = zeros(2*Nnode,1);
u(setdiff(1:2*Nnode,Fixdof)) = u_;

% local displacement
delta = [];
for ii = 1:Nmember
    d_ = [u([MN(ii,1)*2-1+(0:1) MN(ii,2)*2-1+(0:1)])];
    T_ = [T{ii} zeros(2);zeros(2) T{ii}];
    delta = [delta T_*d_];
end

% local nodal force
x = [];
for ii = 1:Nmember
    x = [x kl{ii}*delta(:,ii)];
end

F = -x(1,:)';

end % function end
