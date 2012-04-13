/* BDDCML - Multilevel BDDC
 *
 * This program is a free software.
 * You can redistribute it and/or modify it under the terms of 
 * the GNU Lesser General Public License 
 * as published by the Free Software Foundation, 
 * either version 3 of the license, 
 * or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details
 * <http://www.gnu.org/copyleft/lesser.html>.
 *_______________________________________________________________*/

#include "metis.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

# if defined(UPPER) 
#  define F_SYMBOL(lower_case,upper_case) upper_case
# elif defined(Add_)
#  define F_SYMBOL(lower_case,upper_case) lower_case##_
# elif defined(Add__)
#  define F_SYMBOL(lower_case,upper_case) lower_case##__
# else
#  define F_SYMBOL(lower_case,upper_case) lower_case
# endif

/*****************************************
* Wrapper of METIS_PartGraphRecursive and METIS_PartGraphKWay functions
******************************************/

#define graph_divide_c \
    F_SYMBOL(graph_divide_c,GRAPH_DIVIDE_C)
void graph_divide_c( int *numflag, int *graphtype, int *nvertex, int *xadj, int *lxadj, int *adjncy, int *ladjncy, 
                     int *vwgt, int *lvwgt, int *adjwgt, int *ladjwgt, int *nsub, 
                     int *edgecut, int *part, int *lpart )
{
  /***********************************/
  /* Try and take care of bad inputs */
  /***********************************/
  if (numflag == NULL || graphtype == NULL || nvertex == NULL || 
      xadj == NULL || lxadj == NULL ||
      adjncy == NULL || ladjncy == NULL || vwgt == NULL || lvwgt == NULL || adjwgt == NULL || ladjwgt == NULL ||
      nsub == NULL || edgecut == NULL || part == NULL || lpart == NULL) {
     printf("ERROR in GRAPH_DIVIDE_C: One or more required parameters is NULL. Aborting.\n");
     abort();
  }


  /* prepare options */
  int i;
#if (METIS_VER_MAJOR >= 5)
  int ncon = 1;
  /*int *options = NULL;*/
  int options[METIS_NOPTIONS];
  options[METIS_OPTION_OBJTYPE]   = METIS_OBJTYPE_CUT;
  options[METIS_OPTION_CTYPE]     = METIS_CTYPE_RM;
  options[METIS_OPTION_IPTYPE]    = METIS_IPTYPE_GROW;
  options[METIS_OPTION_RTYPE]     = METIS_RTYPE_GREEDY;
  options[METIS_OPTION_NCUTS]     = 1;
  options[METIS_OPTION_NSEPS]     = 1;
  options[METIS_OPTION_NUMBERING] = *numflag;
  options[METIS_OPTION_NITER]     = 10;
  options[METIS_OPTION_CONTIG]    = 0;
  options[METIS_OPTION_DBGLVL]    = 0;
#else
  /* weights */
  int wgtflag;
  if (*graphtype == 1) {
     wgtflag = 1;
  }
  else {
     wgtflag = 0;
  }
  int options[5];
  for ( i = 0; i < 5; i++ ) {
     options[i] = 0;
  }
#endif

  /* Initialize parts */
  for ( i = 0; i < *lpart; i++ ) {
     part[i] = *numflag;
  }

  /* divide graph */
  if (*nsub == 0) {
     printf("ERROR in GRAPH_DIVIDE_C: Illegal number of subdomains %d,  Aborting.\n", *nsub);
     abort();
  }
  else if (*nsub == 1) {
     *edgecut = 0;
  }
  else if (*nsub > 1 && *nsub <= 8) {

#if (METIS_VER_MAJOR >= 5)
     options[METIS_OPTION_UFACTOR]   = 1;
     METIS_PartGraphRecursive(nvertex,&ncon,xadj,adjncy,NULL,NULL,adjwgt,nsub,NULL,NULL,options,edgecut,part);
#else
     METIS_PartGraphRecursive(nvertex,xadj,adjncy,vwgt,adjwgt,&wgtflag,numflag,nsub,options,edgecut,part);
#endif
  }
  else {
#if (METIS_VER_MAJOR >= 5)
     options[METIS_OPTION_UFACTOR]   = 30;
     METIS_PartGraphKway(nvertex,&ncon,xadj,adjncy,NULL,NULL,adjwgt,nsub,NULL,NULL,options,edgecut,part);
#else
     METIS_PartGraphKway(nvertex,xadj,adjncy,vwgt,adjwgt,&wgtflag,numflag,nsub,options,edgecut,part);
#endif
  }

  return;
}
