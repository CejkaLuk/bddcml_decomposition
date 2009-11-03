module module_graph
! module for operations on graphs
! Jakub Sistek, Praha, 2009

contains 

subroutine graph_get_netn(nelem,inet,linet,nnet,lnnet,netn,lnetn)
! get array of numbers of elements at nodes

implicit none
! number of elements in mesh
integer, intent(in) :: nelem

! PMD mesh description
integer, intent(in) :: linet
integer, intent(in) ::  inet(linet)
integer, intent(in) :: lnnet
integer, intent(in) ::  nnet(lnnet)

! dual mesh desctription 
integer, intent(in) :: lnetn
integer, intent(out) :: netn(lnetn)

! local variables
integer :: ie, indinet, indnode, ine, nne

! Count elements touching particular node
! zero whole field of counts
netn = 0
indinet = 0
do ie = 1,nelem
   nne = nnet(ie)
   do ine = 1,nne
      indinet = indinet + 1
      indnode = inet(indinet)
      netn(indnode) = netn(indnode) + 1
   end do
end do
end subroutine graph_get_netn

subroutine graph_get_ietn(nelem,inet,linet,nnet,lnnet,netn,lnetn,kietn,lkietn,ietn,lietn)
! get array of elements at nodes (pointers are already known kietn)

implicit none
! number of elements in mesh
integer, intent(in) :: nelem

! PMD mesh description
integer, intent(in) :: linet
integer, intent(in) ::  inet(linet)
integer, intent(in) :: lnnet
integer, intent(in) ::  nnet(lnnet)

! dual mesh desctription 
integer, intent(in) :: lkietn
integer, intent(in) ::  kietn(lkietn)
integer, intent(in) :: lnetn
integer, intent(out) :: netn(lnetn)
integer, intent(in) :: lietn
integer, intent(out) :: ietn(lietn)

! local variables
integer :: ie, indinet, indnode, ine, nne, indelmn, pointietn

! Count elements touching particular node
! zero whole field of counts and use it as secondary pointers
      netn = 0
      ietn = 0

      indinet = 0
      do ie = 1,nelem
         nne = nnet(ie)
         do ine = 1,nne
            indinet = indinet + 1
            indnode = inet(indinet)
            netn(indnode) = netn(indnode) + 1
            indelmn   = netn(indnode)
            pointietn = kietn(indnode)
            ietn(pointietn + indelmn) = ie
         end do
      end do
end subroutine graph_get_ietn

subroutine graph_from_mesh_size(nelem,graphtype,neighbouring,inet,linet,nnet,lnnet,ietn,lietn,netn,lnetn,kietn,lkietn,&
                                xadj,lxadj, nedge, ladjncy, ladjwgt)
! find size for graph from mesh in PMD format 
use module_utils
implicit none
! number of elements in mesh
integer, intent(in) :: nelem
! type of output graph
integer, intent(in) :: graphtype
! prescribed value of number of shared nodes between two neighbours
integer, intent(in) :: neighbouring

! PMD mesh description
integer, intent(in) :: linet
integer, intent(in) ::  inet(linet)
integer, intent(in) :: lnnet
integer, intent(in) ::  nnet(lnnet)
! PMD dual mesh description
integer, intent(in) :: lietn
integer, intent(in) ::  ietn(lietn)
integer, intent(in) :: lnetn
integer, intent(in) ::  netn(lnetn)
integer, intent(in) :: lkietn
integer, intent(in) ::  kietn(lkietn)

! METIS graph description
integer, intent(in)  :: lxadj
integer, intent(out) ::  xadj(lxadj)
integer, intent(out) ::  nedge
integer, intent(out) ::  ladjncy
integer, intent(out) ::  ladjwgt

! local variables
integer,allocatable :: onerow(:), onerowweig(:)
integer :: nnetx, netnx, lonerow, lonerowweig, ie, indinet, indnode, ine, ionerow, nelmn, nne, pointietn, lor

! prepare arrays for storing a row
      nnetx = maxval(nnet)
      netnx = maxval(netn)
      lonerow = nnetx * netnx
      lonerowweig = lonerow
      allocate(onerow(lonerow),onerowweig(lonerowweig))
      
      ! zero whole array
      xadj(1) = 1

      indinet = 0
      do ie = 1,nelem
         nne = nnet(ie)
         ! zero local row
         onerow     = 0
         onerowweig = 0
         ionerow = 0
         do ine = 1,nne
            indinet = indinet + 1
            indnode = inet(indinet)
            nelmn   = netn(indnode)
            pointietn = kietn(indnode)
            onerow(ionerow+1:ionerow + nelmn) = ietn(pointietn+1:pointietn+nelmn)
            ionerow = ionerow + nelmn
         end do
         ! parse onerow
         call graph_parse_onerow(ie,neighbouring,onerow,onerowweig,lor)
         xadj(ie + 1) = xadj(ie) + count(onerowweig.ge.neighbouring)
      end do

      ladjncy = xadj(nelem+1)-1
      ! check the graph
      if (mod(ladjncy,2).ne.0) then
         write(*,*) 'GRAPH_FROM_MESH_SIZE: Number of nodes has to be even number!'
         call error_exit
      end if
      nedge = ladjncy / 2
      if (graphtype.eq.1) then
         ladjwgt = ladjncy
      else
         ladjwgt = 0
      end if

      deallocate(onerow,onerowweig)
end subroutine graph_from_mesh_size

subroutine graph_from_mesh(nelem,graphtype,neighbouring,inet,linet,nnet,lnnet,ietn,lietn,netn,lnetn,kietn,lkietn,&
                           xadj,lxadj, adjncy,ladjncy, adjwgt,ladjwgt)
! find size for graph from mesh in PMD format 
use module_utils
implicit none
! number of elements in mesh
integer, intent(in) :: nelem
! type of output graph
integer, intent(in) :: graphtype
! prescribed value of number of shared nodes between two neighbours
integer, intent(in) :: neighbouring

! PMD mesh description
integer, intent(in) :: linet
integer, intent(in) ::  inet(linet)
integer, intent(in) :: lnnet
integer, intent(in) ::  nnet(lnnet)
! PMD dual mesh description
integer, intent(in) :: lietn
integer, intent(in) ::  ietn(lietn)
integer, intent(in) :: lnetn
integer, intent(in) ::  netn(lnetn)
integer, intent(in) :: lkietn
integer, intent(in) ::  kietn(lkietn)

! METIS graph description
integer, intent(in) :: lxadj
integer, intent(in) ::  xadj(lxadj)
integer, intent(in) :: ladjncy
integer, intent(out) :: adjncy(ladjncy)
integer, intent(in) :: ladjwgt
integer, intent(out) :: adjwgt(ladjwgt)

! local variables
integer,allocatable :: onerow(:), onerowweig(:)
integer :: nnetx, netnx, lonerow, lonerowweig, ie, indinet, indnode, ine, ionerow, nelmn, nne, pointietn, lor

! prepare arrays for storing a row
      nnetx = maxval(nnet)
      netnx = maxval(netn)
      lonerow = nnetx * netnx
      lonerowweig = lonerow
      allocate(onerow(lonerow),onerowweig(lonerowweig))
      
      indinet = 0
      do ie = 1,nelem
         onerow     = 0
         onerowweig = 0
         ionerow = 0
         nne = nnet(ie)
         do ine = 1,nne
            indinet = indinet + 1
            indnode = inet(indinet)
            nelmn = netn(indnode)
            pointietn = kietn(indnode)
            onerow(ionerow+1:ionerow + nelmn) = ietn(pointietn+1:pointietn+nelmn)
            ionerow = ionerow + nelmn
         end do
         ! parse onerow
         call graph_parse_onerow(ie,neighbouring,onerow,onerowweig,lor)
         ! now only adjacencies above the level considered
         adjncy(xadj(ie):xadj(ie)+lor-1) = onerow(1:lor)
         if (graphtype.eq.1) then
            adjwgt(xadj(ie):xadj(ie+1)-1) = onerowweig(1:lor)
         end if
      end do

      deallocate(onerow,onerowweig)
end subroutine graph_from_mesh

subroutine graph_check(nvertex,graphtype, xadj,lxadj, adjncy,ladjncy, adjwgt,ladjwgt)
! check the graph
use module_utils
implicit none
! number of vertices in graph
integer, intent(in) :: nvertex
! type of output graph
integer, intent(in) :: graphtype

! METIS graph description
integer, intent(in) :: lxadj
integer, intent(in) ::  xadj(lxadj)
integer, intent(in) :: ladjncy
integer, intent(out) :: adjncy(ladjncy)
integer, intent(in) :: ladjwgt
integer, intent(out) :: adjwgt(ladjwgt)

! local variables
integer :: iadjelm, indadjelm, iadjelmadj, indadjelmadj, ivertex, nadjelm, nadjelmadj
logical :: match

      ! Check the graph
      do ivertex = 1,nvertex
         nadjelm = xadj(ivertex+1) - xadj(ivertex)
         do iadjelm = 1,nadjelm
            indadjelm = adjncy(xadj(ivertex)-1 + iadjelm)
            match = .false. 
            indadjelmadj = 0
            nadjelmadj = xadj(indadjelm+1) - xadj(indadjelm)
            do iadjelmadj = 1,nadjelmadj
               if (adjncy(xadj(indadjelm)-1 + iadjelmadj).eq.ivertex) then
                  if (match) then
                     write(*,*) 'GRAPH_CHECK: Element ',ivertex,' multiply mentioned in the list of neighbours of element', &
                                indadjelm,'!'
                     call error_exit
                  end if
                  match = .true.
                  indadjelmadj = iadjelmadj
               end if
            end do
            if (.not.match) then
               write(*,*) 'GRAPH_CHECK: No match! Couple ',ivertex,indadjelm,' mentioned but not the opposite!'
               call error_exit
            end if
            if (graphtype.eq.1) then
               if (adjwgt(xadj(indadjelm)-1+indadjelmadj).ne.adjwgt(xadj(ivertex)-1+iadjelm)) then
                  write(*,*) 'GRAPH_CHECK: Non-matching edge weights between elements ', ivertex, indadjelm,'!'
!                  write(*,*) 'Indices of adjacent elements:'
!                  do ie2 = 1,nelem
!                     nadjelm = naetet(ie2)
!                     write(*,'(100i7)') ie2, iaetet(ie2,1:nadjelm)
!                  end do
!                  write(*,*) 'Weights of elements:'
!                  do ie2 = 1,nelem
!                     nadjelm = naetet(ie2)
!                     write(*,'(100i7)') ie2, edgeweights(ie2,1:nadjelm)
!                  end do
                  call error_exit
               end if
            end if
         end do
      end do
end subroutine graph_check

subroutine graph_write_to_file(idfile,nvertex,nedge,graphtype, xadj,lxadj, adjncy,ladjncy, adjwgt,ladjwgt)
! Write the list into file for METIS

use module_utils
implicit none
! unit for writing the graph
integer, intent(in) :: idfile
! number of vertices in graph
integer, intent(in) :: nvertex
! number of edges in graph
integer, intent(in) :: nedge
! type of output graph
integer, intent(in) :: graphtype

! METIS graph description
integer, intent(in) :: lxadj
integer, intent(in) ::  xadj(lxadj)
integer, intent(in) :: ladjncy
integer, intent(out) :: adjncy(ladjncy)
integer, intent(in) :: ladjwgt
integer, intent(out) :: adjwgt(ladjwgt)

! local variables
integer :: ie, nadje, j

      write(idfile,'(x,i10,x,i10,x,i10)') nvertex, nedge, graphtype
      do ie = 1,nvertex
         nadje = xadj(ie+1) - xadj(ie)
         if      (graphtype.eq.0) then
            ! no weights 
            write(idfile,'(600i9)') nadje, (adjncy(xadj(ie)-1+j), j = 1,nadje)
         else if (graphtype.eq.1) then
            ! weighted graph 
            write(idfile,'(600i9)') nadje, (adjncy(xadj(ie)-1+j), adjwgt(xadj(ie)-1+j), j = 1,nadje)
         else
            write(*,*) 'GRAPH_WRITE_TO_FILE: Graph type not supported: ',graphtype
            call error_exit
         end if
      end do
end subroutine graph_write_to_file

subroutine graph_parse_onerow(ie,neighbouring,onerow,onerowweig,lor)
implicit none

integer,intent(in) :: ie
integer,intent(in) :: neighbouring
integer,intent(inout) :: onerow(:)
integer,intent(inout) :: onerowweig(:)
integer,intent(inout) :: lor

! local variables
integer :: io, indel

         ! eliminate myself
         where(onerow .eq. ie) onerow = 0
         ! eliminate multiplicities
         lor = count(onerow .ne. 0)
         onerow = pack(onerow,onerow.ne.0)
         onerow(lor+1:) = 0
         io = 0
         do 
            io = io + 1
            indel = onerow(io)
            if (indel.eq.0) exit
            onerowweig(io) = count(onerow.eq.indel)
            where (onerow(io+1:).eq.indel) onerow(io+1:) = 0
            lor = count(onerow .ne. 0)
            onerow = pack(onerow,onerow.ne.0)
            onerow(lor+1:) = 0
         end do
         ! no repeating indices in onerow
         ! eliminate down limit for adjacencies
         where(onerowweig(1:lor).lt.neighbouring) onerow(1:lor)     = 0
         where(onerowweig(1:lor).lt.neighbouring) onerowweig(1:lor) = 0
         lor = count(onerow .ne. 0)
         onerow     = pack(onerow,onerow.ne.0)
         onerowweig = pack(onerowweig,onerowweig.ne.0)
         onerow(lor+1:)     = 0
         onerowweig(lor+1:) = 0

end subroutine graph_parse_onerow


recursive subroutine graph_components(nvertex,xadj,lxadj,adjncy,ladjncy,components,lcomponents,ncomponents)

! find graph components in METIS-like graph

use module_utils
implicit none
integer, intent(in) :: nvertex
integer, intent(in) :: lxadj
integer, intent(in) ::  xadj(lxadj)
integer, intent(in) :: ladjncy
integer, intent(in) ::  adjncy(ladjncy)
integer, intent(in) ::    lcomponents
integer, intent(in out) :: components(lcomponents)
integer, intent(out) ::    ncomponents

! Local variable
integer :: i, icompo

! initialize the components array
components = -1

icompo = 0
do i = 1,nvertex
   if (components(i) .le. 0) then
      icompo = icompo + 1
      call graph_components_1(i)
   end if
   ! debug
   ! print *,'components'
   ! print *,components
end do
ncomponents = icompo 
! check the components
if (any(components.eq.-1)) then
   write(*,*) 'GRAPH_COMPONENTS: Error - some vertices not associated.'
   call error_exit
end if
if (any(components.gt.ncomponents)) then
   write(*,*) 'GRAPH_COMPONENTS: Error - some component indices larger than allowed.'
   call error_exit
end if

contains

recursive subroutine graph_components_1(ivertex)

integer, intent(in) :: ivertex

!     Local variables
integer             :: ineib, indneib

! mark this node into the component
components(ivertex) = icompo

do ineib = xadj(ivertex),xadj(ivertex+1)-1
   indneib = adjncy(ineib)

   if (components(indneib).le.0) then
      call graph_components_1(indneib)
   end if
end do

end subroutine graph_components_1
end subroutine graph_components

subroutine graph_read_dimensions(idgraph,nvertex,nedge,graphtype,lxadj,ladjncy,ladjwgt)
use module_parsing
implicit none
! unit for graph file - supposed open
integer, intent(in) ::  idgraph
integer, intent(out) :: nvertex
integer, intent(out) :: nedge
integer, intent(out) :: graphtype
integer, intent(out) :: lxadj
integer, intent(out) :: ladjncy
integer, intent(out) :: ladjwgt

! read initial line in the file with graph
call rdline(idgraph)
read(line,*) nvertex, nedge, graphtype

! set dimensions for allocation
ladjncy = 2*nedge
if (graphtype.eq.1) then
   ladjwgt = ladjncy
else
   ladjwgt = 0
end if
lxadj   = nvertex + 1
end subroutine graph_read_dimensions

subroutine graph_read(idgraph,nvertex,nedge,graphtype,xadj,lxadj,adjncy,ladjncy,adjwgt,ladjwgt)
! import graph from file - without parsing (fast), no comments allowed
use module_utils
implicit none
! unit for graph file - supposed open
integer, intent(in) ::  idgraph
integer, intent(in) ::  nvertex
integer, intent(in) ::  nedge
integer, intent(in) ::  graphtype

integer, intent(in) ::  lxadj
integer, intent(out) ::  xadj(lxadj)
integer, intent(in) ::  ladjncy
integer, intent(out) ::  adjncy(ladjncy)
integer, intent(in) ::  ladjwgt
integer, intent(out) ::  adjwgt(ladjwgt)

! local variables
integer :: indadjncy, ivertex, j, nneib

indadjncy = 0
xadj(1)   = 1
do ivertex = 1,nvertex
   if      (graphtype.eq.1) then
      read(idgraph,*) nneib,(adjncy(indadjncy+j),adjwgt(indadjncy+j),j = 1,nneib)
   else if (graphtype.eq.0) then
      read(idgraph,*) nneib,(adjncy(indadjncy+j),j = 1,nneib)
   else
      write(*,*) 'GRAPH_READ: Type of graph not supported. graphtype =',graphtype
   end if
   indadjncy = indadjncy + nneib
   xadj(ivertex+1) = xadj(ivertex) + nneib
end do
! check the number of edges read correspond to the prescribed dimension
if (mod(indadjncy,2).ne.0) then
   write(*,*) 'GRAPH_READ: Error in graph - number of connections is odd but must be even.'
   call error_exit
end if
if (indadjncy/2 .ne. nedge) then
   write(*,*) 'GRAPH_READ: Error in graph - number of edges does not match.'
   call error_exit
end if
end subroutine graph_read

subroutine graph_read_parsing(idgraph,nvertex,nedge,graphtype,xadj,lxadj,adjncy,ladjncy,adjwgt,ladjwgt)
! import graph from file - with parsing (slow)
use module_parsing
use module_utils
implicit none
! unit for graph file - supposed open
integer, intent(in) ::  idgraph
integer, intent(in) ::  nvertex
integer, intent(in) ::  nedge
integer, intent(in) ::  graphtype

integer, intent(in) ::  lxadj
integer, intent(out) ::  xadj(lxadj)
integer, intent(in) ::  ladjncy
integer, intent(out) ::  adjncy(ladjncy)
integer, intent(in) ::  ladjwgt
integer, intent(out) ::  adjwgt(ladjwgt)

! local variables
integer :: indadjncy, indvertex, ivertex, nneib

indadjncy = 0
xadj(1)   = 1
do ivertex = 1,nvertex
   call rdline(idgraph)
   indvertex = 0
   call getstring
   read(string,*) nneib
   if (nneib.ne.0) then
      do
         call getstring
         indadjncy = indadjncy + 1
         indvertex = indvertex + 1
         read(string,*) adjncy(indadjncy)
         if (graphtype.eq.1) then
            call getstring
            read(string,*) adjwgt(indadjncy)
         end if
         if (kstring.eq.0) then
            exit
         end if
      end do
   end if
   ! check the line
   if (indvertex .ne. nneib) then
      write(*,*) 'GRAPH_READ_PARSING: Error in graph - number of connections mismatch.'
      write(*,*) 'GRAPH_READ_PARSING: ivertex =',ivertex,'nneib =',nneib,'indvertex =',indvertex
      call error_exit
   end if
   xadj(ivertex+1) = xadj(ivertex) + indvertex
end do
! check the number of edges read correspond to the prescribed dimension
if (mod(indadjncy,2).ne.0) then
   write(*,*) 'GRAPH_READ_PARSING: Error in graph - number of connections is odd but must be even.'
   call error_exit
end if
if (indadjncy/2 .ne. nedge) then
   write(*,*) 'GRAPH_READ_PARSING: Error in graph - number of edges does not match.'
   call error_exit
end if
end subroutine graph_read_parsing

end module module_graph
