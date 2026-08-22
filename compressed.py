#!/bin/sh
_F='pnbrqk'
_E=False
_D='P'
_C=None
_B=True
_A='K'
import time
from itertools import count
from collections import namedtuple
version='sunfish 2026'
piece={_D:100,'N':280,'B':320,'R':479,'Q':929,_A:60000}
pst={_D:(0,0,0,0,0,0,0,0,78,83,86,73,102,82,85,90,7,29,21,44,40,31,44,7,-17,16,-2,15,14,0,15,-13,-26,3,10,9,6,1,0,-23,-22,9,5,-11,-10,-2,3,-19,-31,8,-7,-37,-36,-14,3,-31,0,0,0,0,0,0,0,0),'N':(-66,-53,-75,-75,-10,-55,-58,-70,-3,-6,100,-36,4,62,-4,-14,10,67,1,74,73,27,62,-2,24,24,45,37,33,41,25,17,-1,5,31,21,22,35,2,0,-18,10,13,22,18,15,11,-14,-23,-15,2,0,2,0,-23,-20,-74,-23,-26,-24,-19,-35,-22,-69),'B':(-59,-78,-82,-76,-23,-107,-37,-50,-11,20,35,-42,-39,31,2,-22,-9,39,-32,41,52,-10,28,-14,25,17,20,34,26,25,15,10,13,10,17,23,17,16,0,7,14,25,24,15,8,25,20,15,19,20,11,6,7,6,20,16,-7,2,-15,-12,-14,-15,-10,-10),'R':(35,29,33,4,37,33,56,50,55,29,56,67,55,62,34,60,19,35,28,33,45,27,25,15,0,5,16,13,18,-4,-9,-6,-28,-35,-16,-21,-13,-29,-46,-30,-42,-28,-42,-25,-25,-35,-26,-46,-53,-38,-31,-26,-29,-43,-44,-53,-30,-24,-18,5,-2,-18,-31,-32),'Q':(6,1,-8,-104,69,24,88,26,14,32,60,-10,20,76,57,24,-2,43,32,60,72,63,43,2,1,-16,22,17,25,20,-13,-6,-14,-15,-2,-5,-1,-10,-20,-22,-30,-6,-13,-11,-16,-11,-16,-27,-36,-18,0,-19,-15,-15,-21,-38,-39,-30,-31,-13,-31,-36,-34,-42),_A:(4,54,47,-99,-99,60,83,-62,-32,10,55,56,56,55,10,3,-62,12,-57,44,-67,28,37,-31,-55,50,11,-4,-19,13,0,-49,-55,-43,-52,-28,-51,-47,-8,-50,-47,-42,-43,-79,-64,-32,-29,-32,-4,3,-14,-50,-57,-18,13,4,17,30,-3,-14,6,-1,40,18)}
for(k,table)in pst.items():padrow=lambda row:(0,)+tuple(A+piece[k]for A in row)+(0,);pst[k]=sum((padrow(table[A*8:A*8+8])for A in range(8)),());pst[k]=(0,)*20+pst[k]+(0,)*20
K_MID,K_END=pst[_A],tuple(piece[_A]+70-10*(abs(2*(A//10)-11)+abs(2*(A%10)-9))for A in range(120))
A1,H1,A8,H8=91,98,21,28
initial='         \n         \n rnbqkbnr\n pppppppp\n ........\n ........\n ........\n ........\n PPPPPPPP\n RNBQKBNR\n         \n         \n'
N,E,S,W=-10,1,10,-1
directions={_D:(N,N+N,N+W,N+E),'N':(N+N+E,E+N+E,E+S+E,S+S+E,S+S+W,W+S+W,W+N+W,N+N+W),'B':(N+E,S+E,S+W,N+W),'R':(N,E,S,W),'Q':(N,E,S,W,N+E,S+E,S+W,N+W),_A:(N,E,S,W,N+E,S+E,S+W,N+W)}
MATE_LOWER=piece[_A]-13*piece['Q']
MATE_UPPER=piece[_A]+10*piece['Q']
QS=40
QS_A=140
LMR=75
EVAL_ROUGHNESS=15
NULL_MARGIN=-200
DELAY=200
TABLE_SIZE=10**6
Move=namedtuple('Move','i j prom')
class Position(namedtuple('Position','board score wc bc ep kp')):
	def gen_moves(B):
		for(C,F)in enumerate(B.board):
			if F not in'PNBRQK':continue
			for D in directions[F]:
				for A in count(C+D,D):
					G=B.board[A]
					if G in' \nPNBRQK':break
					if F==_D:
						if D in(N,N+N)and G!='.':break
						if D==N+N and(C<A1+N or B.board[C+N]!='.'):break
						if D in(N+W,N+E)and G=='.'and A!=B.ep and abs(A-B.kp)>1:break
						if A8<=A<=H8:yield from(Move(C,A,B)for B in'NBRQ');break
					yield Move(C,A,'')
					if F in'PNK'or G in _F:break
					if C==A1 and B.board[A+E]==_A and B.wc[0]:yield Move(A+E,A+W,'')
					if C==H1 and B.board[A+W]==_A and B.wc[1]:yield Move(A+W,A+E,'')
	def rotate(A,nullmove=_E):B=nullmove;return Position(A.board[::-1].swapcase(),-A.score,A.bc,A.wc,119-A.ep if A.ep and not B else 0,119-A.kp if A.kp and not B else 0)
	def move(D,move):
		C,B,K=move;I,M=D.board[C],D.board[B];E=lambda board,i,p:board[:i]+p+board[i+1:];A=D.board;F,G,J,H=D.wc,D.bc,0,0;L=D.score+D.value(move);A=E(A,B,A[C]);A=E(A,C,'.');F=F[0]and C!=A1,F[1]and C!=H1;G=G[0]and B!=H8,G[1]and B!=A8
		if I==_A:
			F=_E,_E
			if abs(B-C)==2:H=(C+B)//2;A=E(A,A1 if B<C else H1,'.');A=E(A,H,'R')
		if I==_D:
			if A8<=B<=H8:A=E(A,B,K)
			if B-C==2*N:J=C+N
			if B==D.ep:A=E(A,B+S,'.')
		return Position(A,L,F,G,J,H).rotate()
	def value(D,move):
		C,A,G=move;E,F=D.board[C],D.board[A];B=pst[E][A]-pst[E][C]
		if F in _F:B+=pst[F.upper()][119-A]
		if abs(A-D.kp)<2:B+=pst[_A][119-A]
		if E==_A and abs(C-A)==2:B+=pst['R'][(C+A)//2];B-=pst['R'][A1 if A<C else H1]
		if E==_D:
			if A8<=A<=H8:B+=pst[G][A]-pst[_D][A]
			if A==D.ep:B+=pst[_D][119-(A+S)]
		return B
	def king_capture(A):return next((B for B in A.gen_moves()if A.board[B.j]=='k'or abs(B.j-A.kp)<2),_C)
class Stop(Exception):0
Entry=namedtuple('Entry','lower upper')
class Searcher:
	def __init__(A):A.tp_score,A.tp_move,A.history={},{},set();A.nodes,A.deadline,A.soft=0,1<<63,1<<63
	def bound(C,pos,gamma,depth,root=_E):
		K=root;D=gamma;B=depth;A=pos;C.nodes+=1
		if C.nodes%2048==0 and time.time()>C.deadline:raise Stop
		B=max(B,0)
		if A.score<=-MATE_LOWER:return-MATE_UPPER
		if not K:
			G=C.tp_score.get((A,B),Entry(-MATE_UPPER,MATE_UPPER))
			if G.lower>=D:return G.lower
			if G.upper<D:return G.upper
			if B>0 and A in C.history:return 0
		L=C.tp_move.get(A)
		def Q():
			if 2<B<6 and O:yield(_C,_C)
			if B==0:yield(_C,_C)
			if L and((C:=A.value(L))>=QS or B)and(C>=MATE_LOWER or B>3 or A.score+C+max(B-1,0)*QS_A>=D):yield(C,L)
			yield from sorted(((E,C)for C in A.gen_moves()if(E:=A.value(C))>=QS or B),reverse=_B)
		N=abs(A.score)<750 and any(B in A.board for B in'RBNQ');O=not K and N;P=A.score+NULL_MARGIN;R=N and B>=6 and-C.bound(A.rotate(nullmove=_B),1-P,B-7)>=P;E,J=-MATE_UPPER,_E
		for(M,H)in Q():
			if H is _C and B==0:F=A.score
			elif H is _C:
				if(I:=A.score+EVAL_ROUGHNESS)>=D:
					F=min(I,-C.bound(A.rotate(nullmove=_B),1-D,B-3))
					if F>=D and(S:=A.king_capture()):H,F,J=S,MATE_UPPER,_B
				else:F=I
			elif M>=MATE_LOWER:F=MATE_UPPER;J=_B
			else:
				I=MATE_UPPER if B>3 else A.score+M+max(B-1,0)*QS_A
				if I<D:E=max(E,I);break
				T=B-1-(O and B>=6 and M<LMR)-int(R);F=min(I,-C.bound(A.move(H),1-D,T));J|=F>-MATE_UPPER
			E=max(E,F)
			if E>=D:
				if H is not _C and B:
					C.tp_move[A]=H
					if len(C.tp_move)>TABLE_SIZE:del C.tp_move[next(A for A in C.tp_move if A!=C.root)]
				break
		if B and not J and all(A.move(B).king_capture()for B in A.gen_moves()):U=max(1-MATE_UPPER,-MATE_LOWER-B*EVAL_ROUGHNESS);E=U if A.rotate(nullmove=_B).king_capture()else 0
		if not K:C.tp_score[A,B]=Entry(E,G.upper)if E>=D else Entry(G.lower,E)
		if len(C.tp_score)>TABLE_SIZE:del C.tp_score[next(iter(C.tp_score))]
		return E
	def search(A,history):
		G=history;A.nodes,A.history=0,set(G);A.tp_score.clear();D=A.root=G[-1];pst[_A]=K_MID if'Q'in D.board and'q'in D.board else K_END;B=0
		for H in range(1,1000):
			E,F=1-MATE_UPPER,MATE_UPPER
			while E<F-EVAL_ROUGHNESS:
				C=A.bound(D,B,H,root=_B)
				if C>=B:E=C
				if C<B:F=C
				yield(H,B,C,A.tp_move.get(D));B=(E+F+1)//2
			if time.time()>A.soft:return
def parse(c):return A1+ord(c[0])-ord('a')-10*(int(c[1])-1)
def render(i):return chr((i-A1)%10+ord('a'))+str(1-(i-A1)//10)
hist=[Position(initial,0,(_B,_B),(_B,_B),0,0)]
def main():
	Q='score cp';P='info depth';H=Searcher()
	while _B:
		A=input().split()
		if A[0]=='uci':print('id name',version);print('uciok')
		elif A[0]=='isready':print('readyok')
		elif A[0]=='quit':break
		elif A[:2]==['position','startpos']:
			del hist[1:]
			for(R,B)in enumerate(A[3:]):
				C,D,S=parse(B[:2]),parse(B[2:4]),B[4:].upper()
				if R%2==1:C,D=119-C,119-D
				hist.append(hist[-1].move(Move(C,D,S)))
		elif A[0]=='go':
			K=dict(zip(A[1::2],map(int,A[2::2])));L='wb'[len(hist)%2==0];I,T=K.get(L+'time',60000),K.get(L+'inc',0);M=I/40+T-DELAY;U=max(min(M,I/4-DELAY),100)/1000;V=max(min(5*M,I/2-DELAY),200)/1000;N=time.time();H.deadline,H.soft=N+V,N+U;F,E,O=_C,_C,1
			try:
				for(G,W,J,B)in H.search(hist):
					if G>O:F,O=E or F,G
					if J>=W:
						if B is _C:print(P,G,Q,J);break
						C,D=B.i,B.j
						if len(hist)%2==0:C,D=119-C,119-D
						E=render(C)+render(D)+B.prom.lower();print(P,G,Q,J,'pv',E)
			except Stop:E=F or E
			print('bestmove',E or F or'(none)')
if __name__=='__main__':main()