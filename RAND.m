function rand_ind=RAND(X,Y)
a=0;
b=0;
c=0;
d=0;
rand_ind=-1;

if (size(X,1)==size(Y,1))
	for i=1:size(X,1)
	for j=1:size(Y,1)
		if (i~=j)
			if(X(i)==X(j) && Y(i)==Y(j)) a=a+1;
			end
			if(X(i)~=X(j) && Y(i)~=Y(j)) b=b+1;
			end
			if(X(i)==X(j) && Y(i)~=Y(j)) c=c+1;
			end
			if(X(i)~=X(j) && Y(i)==Y(j)) d=d+1;
			end
		end
	end
	end
	rand_ind=(a+b)/(a+b+c+d);
else display('size mismatch');
end
