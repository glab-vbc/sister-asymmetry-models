#!/usr/bin/env python
# coding: utf-8

import logging

import os, sys, copy, re

import numpy as np
from numpy.random import default_rng

from math import ceil, sqrt

import pathlib
import pprint
import string

import gsd
import gsd.hoomd

import freud

import hoomd
import hoomd.md
import hoomd.md.integrate
   
import scipy.stats
 
import bisect

import datetime
import time

################################### FUNCTIONS TO MOVE TO LIBRARY #########################################

def exponentially_separated_links(
    chain_length:int, 
    frequency:float, 
    start_position=0, 
    end_position=None,
    add_boundaries=False,
    min_separation=1,
):

    assert frequency >= 0 and frequency <= 1, "Frequency of links=1/avg_separation should be comprised between 0 and 1."
    
    if end_position is None:
        end_position = chain_length
        
    links_positions = []
    
    rng = np.random.default_rng()
    
    last_link_position = -1
    
    while last_link_position <= (chain_length-1):
        next_link_position = int(last_link_position + min_separation + np.round(rng.exponential(scale=(1/frequency)-min_separation))) 
        
        if (next_link_position <= (chain_length-1)):
            links_positions.append(next_link_position)
        
        last_link_position = next_link_position
    
    links_positions = np.column_stack((links_positions, links_positions))
            
    '''
    for position in range(start_position, end_position):
        n = rng.random()
        if n < frequency:
            links_positions.append((position,position))
    '''
    
    if add_boundaries:
        if not np.array_equal(links_positions[0], [0,0]):
            links_positions = np.vstack([[0,0], links_positions]) #add fixed links at chains extremities that work as boundaries
        if not np.array_equal(links_positions[-1], [chain_length-1,chain_length-1]):
            links_positions = np.vstack([links_positions, [chain_length-1, chain_length-1]])
            
    return links_positions


def exponentially_separated_link_pairs(
    chain_length,
    frequency_intrapair_links,
    frequency_interpair_links,
    min_separation=1,
    add_boundaries=False,
):
    
    assert frequency_intrapair_links >= 0 and frequency_intrapair_links <= 1, "Frequency of links(=1/avg_separation) belonging to the same pair should be comprised between 0 and 1."
    assert frequency_interpair_links >= 0 and frequency_interpair_links <= 1, "Frequency of links(=1/avg_separation) belonging to two consecutive pairs should be comprised between 0 and 1."
    assert frequency_interpair_links <= frequency_intrapair_links, "Frequency of links(=1/avg_separation) belonging to two consecutive pairs should be lower or equal than frequency of links belonging to the same pair."
    
    links_positions = []
    pair_ids = []
    last_link_position = 0
    pair_id = 0
    
    rng = np.random.default_rng()
    
    while last_link_position <= (chain_length-1):
        next_link_position_next_pair = int(last_link_position + min_separation + np.round(rng.exponential(scale=(1/frequency_interpair_links)-min_separation)))        
        next_link_position_same_pair = int(next_link_position_next_pair + min_separation + np.round(rng.exponential(scale=(1/frequency_intrapair_links)-min_separation)))
        
        if (next_link_position_next_pair <= (chain_length-1)) and (next_link_position_same_pair <= (chain_length-1)):
            links_positions.extend([next_link_position_next_pair, next_link_position_same_pair])
            pair_ids.extend([pair_id,pair_id])
        
        last_link_position = next_link_position_same_pair
        pair_id += 1
    
    links_positions = np.column_stack((links_positions, links_positions))
    
    if add_boundaries:
        if not np.array_equal(links_positions[0], [0,0]):
            links_positions = np.vstack([[0,0], links_positions]) #add fixed links at chains extremities that work as boundaries
        if not np.array_equal(links_positions[-1], [chain_length-1,chain_length-1]):
            links_positions = np.vstack([links_positions, [chain_length-1, chain_length-1]])
        pair_ids.insert(0, -1)
        pair_ids.append(-1)
    
    return links_positions, pair_ids


def exponentially_separated_semianchored_links(
    chain_length,
    frequency_intrapair_links,
    frequency_interpair_links,
    min_separation=1,
    nb_links_same_anchor=1,
    add_boundaries=False,
):
    """
    Generate exponentially separated links on a chain with a specified number of consecutive links 
    anchored on the same chain before adding the next pair.
    
    Parameters:
    - chain_length: int, length of the chain.
    - frequency_intrapair_links: float, frequency of links within the same anchor group.
    - frequency_interpair_links: float, frequency of links between pairs.
    - min_separation: int, minimum separation between links (default: 1).
    - nb_links_same_anchor: int, number of consecutive links anchored on the same chain.
    - add_boundaries: bool, whether to add boundary links at 0 and chain_length - 1 (default: False).
    
    Returns:
    - links_positions: np.ndarray, positions of the links as a 2D array.
    - pair_ids: list, IDs for the pairs of links.
    """
    assert 0 <= frequency_intrapair_links <= 1, \
        "Frequency of links (1/avg_separation) belonging to the same pair should be between 0 and 1."
    assert 0 <= frequency_interpair_links <= 1, \
        "Frequency of links (1/avg_separation) belonging to two consecutive pairs should be between 0 and 1."
    assert frequency_interpair_links <= frequency_intrapair_links, \
        "Frequency of links (1/avg_separation) between pairs should be <= frequency within pairs."
    
    
    links_positions = []
    pair_ids = []
    last_link_position = 0
    pair_id = 0
    
    rng = np.random.default_rng()
    
    while last_link_position <= (chain_length-1):
        
        chunk_links_positions = []
        
        # Calculate the position of the first link in the next pair
        next_link_position_next_pair = int(last_link_position + min_separation + 
                                           np.round(rng.exponential(scale=(1/frequency_interpair_links)-min_separation)))
        chunk_links_positions.append(next_link_position_next_pair)
        
        # Generate the `nb_links_same_anchor` links for the current anchor group
        for _ in range((nb_links_same_anchor*2)-1):
            next_link_position_same_pair = int(
                chunk_links_positions[-1] + min_separation + 
                np.round(rng.exponential(scale=(1/frequency_intrapair_links) - min_separation))
            )
            chunk_links_positions.append(next_link_position_same_pair)
        
        if all(link_pos <= (chain_length-1) for link_pos in chunk_links_positions):
            links_positions.append(chunk_links_positions)
            pair_ids.append([pair_id]*len(chunk_links_positions))
        
        last_link_position = chunk_links_positions[-1]
        pair_id += 1
    
    links_positions = np.array(links_positions).flatten()
    pair_ids = np.array(pair_ids).flatten().tolist()
    
    links_positions = np.column_stack((links_positions, links_positions))
    
    if add_boundaries:
        if not np.array_equal(links_positions[0], [0,0]):
            links_positions = np.vstack([[0,0], links_positions]) #add fixed links at chains extremities that work as boundaries
        if not np.array_equal(links_positions[-1], [chain_length-1,chain_length-1]):
            links_positions = np.vstack([links_positions, [chain_length-1, chain_length-1]])
        pair_ids.insert(0, -1)
        pair_ids.append(-1)
    
    return links_positions, pair_ids
    


def set_sliding_links_configuration(
    links,
    nb_fixed_sides,
    fixed_sides_conf,
    chain_first_anchor=0,
    set_boundaries=False,
    **kwargs,
):
        
    if nb_fixed_sides == 2:
        sliding_links_configuration = np.zeros_like(links)
        
    elif nb_fixed_sides == 0:
        sliding_links_configuration = np.ones_like(links)
        if set_boundaries:
            sliding_links_configuration[0] = 0
            sliding_links_configuration[-1] = 0
            
    elif nb_fixed_sides == 1:
        
        if chain_first_anchor == None:
                chain_first_anchor = 0
                
        if fixed_sides_conf == "random":
            rng = np.random.default_rng()
            sliding_links_configuration = np.ones_like(links)
            sliding_links_configuration[np.arange(sliding_links_configuration.shape[0]), rng.integers(2, size=sliding_links_configuration.shape[0])] = 0
            if set_boundaries:
                sliding_links_configuration[0] = 0
                sliding_links_configuration[-1] = 0
                
        elif fixed_sides_conf == "same":
            sliding_links_configuration = np.ones_like(links)
            sliding_links_configuration[:,chain_first_anchor] = 0
            if set_boundaries:
                sliding_links_configuration[0] = 0
                sliding_links_configuration[-1] = 0
                
        elif fixed_sides_conf == "opposite":
            nb_same_anchor = kwargs.get('nb_same_anchor', 1)
            sliding_links_configuration = np.ones_like(links)
            if set_boundaries:
                sliding_links_configuration[0] = 0
                sliding_links_configuration[-1] = 0
                sliding_links_configuration[1:-1,chain_first_anchor] = np.arange(sliding_links_configuration.shape[0]-2)//nb_same_anchor%2
                sliding_links_configuration[1:-1,abs(chain_first_anchor-1)] = ((np.arange(sliding_links_configuration.shape[0]-2)//nb_same_anchor)+1)%2
            else:
                sliding_links_configuration[:,chain_first_anchor] = np.arange(len(sliding_links_configuration))//nb_same_anchor%2
                sliding_links_configuration[:,abs(chain_first_anchor-1)] = ((np.arange(len(sliding_links_configuration))//nb_same_anchor)+1)%2
                
        else:
            raise ValueError('Unrecognized "fixed_sides_conf" value. Accepted values are: "random"|"same"|"opposite". "opposite" has an optional param "nb_same_anchor" defining after how many links the fixed side is set on the opposite side. Default value "nb_same_anchor" is 1.')

    else:
        raise ValueError('Unrecognized "nb_fixed_sides" value. Accepted values are: 0|1|2')

    return sliding_links_configuration


def lattice_SAW_2D(N, t, step_length=1):
    
    from scipy.spatial.distance import cdist

    # initial configuration. Usually we just use a straight chain as inital configuration
    init_state = np.dstack((np.arange(N),np.zeros(N)))[0]
    state = init_state.copy()

    # define a rotation matrix
    rotate_matrix = np.array([[[1,0],[0,1]], 
                               [[0,-1],[1,0]],
                               [[-1,0],[0,-1]],
                               [[0,1],[-1,0]],
                               [[1,0],[0,-1]],
                               [[0,1],[1,0]],
                               [[-1,0],[0,1]],
                               [[0,-1],[-1,0]]]
                             )
    
    # define a dot product function used for the rotate operation
    def v_dot(a):return lambda b: np.dot(a,b)

    # define pivot algorithm process where t is the number of successful steps
    def walk(state, step_length, rotate_matrix, t):
        acpt = 0
        # while loop until the number of successful step up to t
        while acpt <= t:
            pick_pivot = np.random.randint(1,N-1) # pick a pivot site
            pick_side = np.random.choice([-1,1]) # pick a side

            if pick_side == 1:
                old_chain = state[0:pick_pivot+1]
                temp_chain = state[pick_pivot+1:]
            else:
                old_chain = state[pick_pivot:]
                temp_chain = state[0:pick_pivot]

            # pick a symmetry operator
            symtry_oprtr = rotate_matrix[np.random.randint(len(rotate_matrix))]
            # new chain after symmetry operator
            new_chain = np.apply_along_axis(v_dot(symtry_oprtr),1,temp_chain - state[pick_pivot]) + state[pick_pivot]

            # use cdist function of scipy package to calculate the pair-pair distance between old_chain and new_chain
            overlap = cdist(new_chain,old_chain)
            overlap = overlap.flatten()

            # determinte whether the new state is accepted or rejected
            if len(np.nonzero(overlap)[0]) != len(overlap):
                continue
            else:
                if pick_side == 1:
                    state = np.concatenate((old_chain,new_chain),axis=0)
                elif pick_side == -1:
                    state = np.concatenate((new_chain,old_chain),axis=0)
                acpt += 1

        # place the center of mass of the chain on the origin
        state = step_length*(state - np.int_(np.mean(state,axis=0)))
        
        return state
    
        
    final_state = walk(state, step_length, rotate_matrix, t)
    
    return final_state


def cohesed_sisters_starting_configuration_new(sister_length, cohesion_bonds, step=1.0, top_chain=0, bottom_chain=1):
    
    positions = np.zeros((sister_length*2,3))
    
    top_z = 1
    bottom_z = 0
    
    # If a chain does not start from zero, shift its start to zero 
    sister_shift_top = np.sign(max(cohesion_bonds[0,top_chain] + 1 - sister_length, 0)) * sister_length
    sister_shift_bottom = np.sign(max(cohesion_bonds[0,bottom_chain] + 1 - sister_length, 0)) * sister_length
    cohesion_bonds_shifted = np.copy(cohesion_bonds)
    cohesion_bonds_shifted[:,top_chain] -= sister_shift_top
    cohesion_bonds_shifted[:,bottom_chain] -= sister_shift_bottom
    
    # Generate a self-avoiding walk of the length of the backbone (without loops between cohesion bonds)
    # i.e. backbone = nb_cohesion_bonds + chunk before first cohesion bond + chunk after last cohesion bond
    nb_sites_before_first_bond = max(cohesion_bonds_shifted[0,top_chain], cohesion_bonds_shifted[0,bottom_chain])
    nb_sites_after_last_bond = max(sister_length-1-cohesion_bonds_shifted[-1,top_chain], sister_length-1-cohesion_bonds_shifted[-1,bottom_chain])
    N_saw = cohesion_bonds_shifted.shape[0] + nb_sites_before_first_bond + nb_sites_after_last_bond
            
    successful_pivot_steps = 100 #N_saw * 10
    saw = lattice_SAW_2D(N_saw,successful_pivot_steps)
    
    # Assign positions of the generated SAW to cohesion bond positions
    positions[cohesion_bonds[:,top_chain]] = np.column_stack((saw[nb_sites_before_first_bond:nb_sites_before_first_bond+len(cohesion_bonds_shifted)], np.zeros(len(saw[nb_sites_before_first_bond:nb_sites_before_first_bond+len(cohesion_bonds_shifted)]))+top_z))
    positions[cohesion_bonds[:,bottom_chain]] = np.column_stack((saw[nb_sites_before_first_bond:nb_sites_before_first_bond+len(cohesion_bonds_shifted)], np.zeros(len(saw[nb_sites_before_first_bond:nb_sites_before_first_bond+len(cohesion_bonds_shifted)]))+bottom_z))

    # Assign positions of the generated SAW to sites before the first cohesion bond
    positions[sister_shift_top:cohesion_bonds[0,top_chain]] = np.column_stack((saw[(nb_sites_before_first_bond-cohesion_bonds_shifted[0,top_chain]):nb_sites_before_first_bond], np.zeros(len(saw[(nb_sites_before_first_bond-cohesion_bonds_shifted[0,top_chain]):nb_sites_before_first_bond]))+top_z))
    positions[sister_shift_bottom:cohesion_bonds[0,bottom_chain]] = np.column_stack((saw[(nb_sites_before_first_bond-cohesion_bonds_shifted[0,bottom_chain]):nb_sites_before_first_bond], np.zeros(len(saw[(nb_sites_before_first_bond-cohesion_bonds_shifted[0,bottom_chain]):nb_sites_before_first_bond]))+bottom_z))

    # Assign positions of the generated SAW to sites after the last cohesion bond
    positions[(cohesion_bonds[-1,top_chain]+1):(sister_length+sister_shift_top)] = np.column_stack((saw[(nb_sites_before_first_bond+len(cohesion_bonds_shifted)):(nb_sites_before_first_bond+len(cohesion_bonds_shifted)+(sister_length-1-cohesion_bonds_shifted[-1,top_chain]))], np.zeros(len(saw[(nb_sites_before_first_bond+len(cohesion_bonds_shifted)):(nb_sites_before_first_bond+len(cohesion_bonds_shifted)+(sister_length-1-cohesion_bonds_shifted[-1,top_chain]))]))+top_z))
    positions[(cohesion_bonds[-1,bottom_chain]+1):(sister_length+sister_shift_bottom)] = np.column_stack((saw[(nb_sites_before_first_bond+len(cohesion_bonds_shifted)):(nb_sites_before_first_bond+len(cohesion_bonds_shifted)+(sister_length-1-cohesion_bonds_shifted[-1,bottom_chain]))], np.zeros(len(saw[(nb_sites_before_first_bond+len(cohesion_bonds_shifted)):(nb_sites_before_first_bond+len(cohesion_bonds_shifted)+(sister_length-1-cohesion_bonds_shifted[-1,bottom_chain]))]))+bottom_z))

    for i in range(len(cohesion_bonds)-1):
        
        sites_between_consecutive_bonds_top = list(np.arange(cohesion_bonds[i,top_chain]+1, cohesion_bonds[i+1,top_chain]))
        sites_between_consecutive_bonds_bottom = list(np.arange(cohesion_bonds[i,bottom_chain]+1, cohesion_bonds[i+1,bottom_chain]))
        
        if len(sites_between_consecutive_bonds_top)>=2:
            site_1 = sites_between_consecutive_bonds_top.pop(0)
            site_2 = sites_between_consecutive_bonds_top.pop(-1)
            positions[site_1] = (positions[site_1-1][0]+0.25, positions[site_1-1][1], positions[site_1-1][2]+1)
            positions[site_2] = (positions[site_2+1][0]-0.25, positions[site_2+1][1], positions[site_2+1][2]+1)
            while len(sites_between_consecutive_bonds_top)>=2:
                site_1 = sites_between_consecutive_bonds_top.pop(0)
                site_2 = sites_between_consecutive_bonds_top.pop(-1)
                positions[site_1] = (positions[site_1-1][0], positions[site_1-1][1], positions[site_1-1][2]+1)
                positions[site_2] = (positions[site_2+1][0], positions[site_2+1][1], positions[site_2+1][2]+1)              
        if len(sites_between_consecutive_bonds_top) == 1:
            last_site = sites_between_consecutive_bonds_top.pop(0)
            last_site_x = (positions[last_site-1][0] + positions[last_site+1][0]) / 2
            last_site_y = (positions[last_site-1][1] + positions[last_site+1][1]) / 2
            last_site_z = positions[last_site-1][2] + 1
            
            positions[last_site] = (last_site_x, last_site_y, last_site_z)
            
        if len(sites_between_consecutive_bonds_bottom)>=2:
            site_1 = sites_between_consecutive_bonds_bottom.pop(0)
            site_2 = sites_between_consecutive_bonds_bottom.pop(-1)
            positions[site_1] = (positions[site_1-1][0]+0.25, positions[site_1-1][1], positions[site_1-1][2]-1)
            positions[site_2] = (positions[site_2+1][0]-0.25, positions[site_2+1][1], positions[site_2+1][2]-1)
            while len(sites_between_consecutive_bonds_bottom)>=2:
                site_1 = sites_between_consecutive_bonds_bottom.pop(0)
                site_2 = sites_between_consecutive_bonds_bottom.pop(-1)
                positions[site_1] = (positions[site_1-1][0], positions[site_1-1][1], positions[site_1-1][2]-1)
                positions[site_2] = (positions[site_2+1][0], positions[site_2+1][1], positions[site_2+1][2]-1)
        if len(sites_between_consecutive_bonds_bottom) == 1:
            last_site = sites_between_consecutive_bonds_bottom.pop(0)
            last_site_x = (positions[last_site-1][0] + positions[last_site+1][0]) / 2
            last_site_y = (positions[last_site-1][1] + positions[last_site+1][1]) / 2
            last_site_z = positions[last_site-1][2] - 1
            positions[last_site] = (last_site_x, last_site_y, last_site_z)
            
    return positions

def cohesed_sisters_starting_configuration(sister_length, cohesion_bonds, init_point=(0,0,0), step=1.0, top_chain=0, bottom_chain=1):
    
    positions = np.zeros((sister_length*2,3))
    
    top_y = 1
    bottom_y = 0
    
    # If a chain does not start from zero, shift its start to zero 
    sister_shift_top = np.sign(max(cohesion_bonds[0,top_chain] + 1 - sister_length, 0)) * sister_length
    sister_shift_bottom = np.sign(max(cohesion_bonds[0,bottom_chain] + 1 - sister_length, 0)) * sister_length

    first_avail_position = 0
    nb_sites_before_first_cohes_bond_top = cohesion_bonds[0,top_chain] - sister_shift_top
    nb_sites_before_first_cohes_bond_bottom = cohesion_bonds[0,bottom_chain] - sister_shift_bottom
    if (nb_sites_before_first_cohes_bond_top > 0) or (nb_sites_before_first_cohes_bond_bottom > 0):
        positions[sister_shift_top:cohesion_bonds[0,top_chain]][::-1] = np.column_stack(([first_avail_position]*nb_sites_before_first_cohes_bond_top, np.arange(nb_sites_before_first_cohes_bond_top)+top_y, [0]*nb_sites_before_first_cohes_bond_top)) + init_point
        positions[sister_shift_bottom:cohesion_bonds[0,bottom_chain]][::-1] = np.column_stack(([first_avail_position]*nb_sites_before_first_cohes_bond_bottom, (np.arange(nb_sites_before_first_cohes_bond_bottom)+bottom_y)*(-1), [0]*nb_sites_before_first_cohes_bond_bottom)) + init_point
        first_avail_position += 1
        #print(f"nb_sites_before_first_cohes_bond_top {nb_sites_before_first_cohes_bond_top}")    
        #print(f"nb_sites_before_first_cohes_bond_bottom {nb_sites_before_first_cohes_bond_bottom}")  
        #print(positions)
    
    positions[cohesion_bonds[:,top_chain]] = np.column_stack((np.arange(first_avail_position, cohesion_bonds.shape[0]+first_avail_position), [top_y]*cohesion_bonds.shape[0], [0]*cohesion_bonds.shape[0]))
    positions[cohesion_bonds[:,bottom_chain]] = np.column_stack((np.arange(first_avail_position, cohesion_bonds.shape[0]+first_avail_position), [bottom_y]*cohesion_bonds.shape[0], [0]*cohesion_bonds.shape[0]))
    
    for i in range(len(cohesion_bonds)-1):
        
        sites_between_consecutive_bonds_top = list(np.arange(cohesion_bonds[i,top_chain]+1, cohesion_bonds[i+1,top_chain]))
        sites_between_consecutive_bonds_bottom = list(np.arange(cohesion_bonds[i,bottom_chain]+1, cohesion_bonds[i+1,bottom_chain]))
        
        if len(sites_between_consecutive_bonds_top)>=2:
            site_1 = sites_between_consecutive_bonds_top.pop(0)
            site_2 = sites_between_consecutive_bonds_top.pop(-1)
            positions[site_1] = (positions[site_1-1][0]+0.25, positions[site_1-1][1]+1, positions[site_1-1][2])
            positions[site_2] = (positions[site_2+1][0]-0.25, positions[site_2+1][1]+1, positions[site_2+1][2])
            while len(sites_between_consecutive_bonds_top)>=2:
                site_1 = sites_between_consecutive_bonds_top.pop(0)
                site_2 = sites_between_consecutive_bonds_top.pop(-1)
                positions[site_1] = (positions[site_1-1][0], positions[site_1-1][1]+1, positions[site_1-1][2])
                positions[site_2] = (positions[site_2+1][0], positions[site_2+1][1]+1, positions[site_2+1][2])              
        if len(sites_between_consecutive_bonds_top) == 1:
            last_site = sites_between_consecutive_bonds_top.pop(0)
            last_site_x = (positions[last_site-1][0] + positions[last_site+1][0] + np.sqrt(3)*(positions[last_site-1][1]-positions[last_site+1][1]))/2 # This computes the `x coordinate` of the third vertex.
            last_site_y = (positions[last_site+1][1] + positions[last_site-1][1] + np.sqrt(3)*(positions[last_site+1][0]-positions[last_site-1][0]))/2 #This computes the 'y coordinate' of the third vertex.
            last_site_z = positions[last_site-1][2]
            positions[last_site] = (last_site_x, last_site_y, last_site_z)
            
        if len(sites_between_consecutive_bonds_bottom)>=2:
            site_1 = sites_between_consecutive_bonds_bottom.pop(0)
            site_2 = sites_between_consecutive_bonds_bottom.pop(-1)
            positions[site_1] = (positions[site_1-1][0]+0.25, positions[site_1-1][1]-1, positions[site_1-1][2])
            positions[site_2] = (positions[site_2+1][0]-0.25, positions[site_2+1][1]-1, positions[site_2+1][2])
            while len(sites_between_consecutive_bonds_bottom)>=2:
                site_1 = sites_between_consecutive_bonds_bottom.pop(0)
                site_2 = sites_between_consecutive_bonds_bottom.pop(-1)
                positions[site_1] = (positions[site_1-1][0], positions[site_1-1][1]-1, positions[site_1-1][2])
                positions[site_2] = (positions[site_2+1][0], positions[site_2+1][1]-1, positions[site_2+1][2])
        if len(sites_between_consecutive_bonds_bottom) == 1:
            last_site = sites_between_consecutive_bonds_bottom.pop(0)
            last_site_x = (positions[last_site-1][0] + positions[last_site+1][0] + np.sqrt(3)*(positions[last_site-1][1]-positions[last_site+1][1]))/2 # This computes the `x coordinate` of the third vertex.
            last_site_y = (positions[last_site-1][1] + positions[last_site+1][1] + np.sqrt(3)*(positions[last_site-1][0]-positions[last_site+1][0]))/2 #This computes the 'y coordinate' of the third vertex.
            last_site_z = positions[last_site-1][2]
            positions[last_site] = (last_site_x, last_site_y, last_site_z)
            
        first_avail_position += 1
    
    nb_sites_after_last_cohes_bond_top = sister_length - ((cohesion_bonds[-1,top_chain]+1) - sister_shift_top)
    nb_sites_after_last_cohes_bond_bottom = sister_length - ((cohesion_bonds[-1,bottom_chain]+1) - sister_shift_bottom)
    positions[(cohesion_bonds[-1,top_chain]+1):(sister_length+sister_shift_top)] = np.column_stack(([cohesion_bonds.shape[0]+1]*nb_sites_after_last_cohes_bond_top, np.arange(nb_sites_after_last_cohes_bond_top)+top_y, [0]*nb_sites_after_last_cohes_bond_top)) + init_point
    positions[(cohesion_bonds[-1,bottom_chain]+1):(sister_length+sister_shift_bottom)] = np.column_stack(([cohesion_bonds.shape[0]+1]*nb_sites_after_last_cohes_bond_bottom, (np.arange(nb_sites_after_last_cohes_bond_bottom)+bottom_y)*(-1), [0]*nb_sites_after_last_cohes_bond_bottom)) + init_point
    
    return positions


def convert_misalignment_distribution_params(distribution_name, distribution_params_bp, bp_monomer=200):
   
    supported_distributions = ["norm", "expon", "exponnorm", "laplace_asymmetric"] #,"lognorm" and "linspace" to implement
   
    assert distribution_name in supported_distributions, f'Unsupported scipy distribution {distribution_name} for parameter conversion. Supported distributions are {supported_distributions}'
   
    distribution_params = dict()
       
    if distribution_name == "norm":
        
        required_params =  set(["scale_bp",])
        optional_params =  set(["loc_bp",])
        
        if not required_params.issubset(distribution_params_bp.keys()):
            raise Exception(f"Values for {required_params} must be specified with misalignment distribution={distribution_name}.")
        if not set(distribution_params_bp.keys() - required_params).issubset(optional_params):
            raise Exception(f"Optional parameters accepted for misalignment distribution={distribution_name} are: {optional_params}.")
            
        distribution_params["scale"] = round(distribution_params_bp["scale_bp"] / bp_monomer)
        
        if "loc_bp" in distribution_params_bp:
            distribution_params["loc"] = round(distribution_params_bp["loc_bp"] / bp_monomer)
   
    if distribution_name == "expon":
        
        required_params =  set(["scale_bp",])
        optional_params =  set(["loc_bp",])
        
        if not required_params.issubset(distribution_params_bp.keys()):
            raise Exception(f"Values for {required_params} must be specified with misalignment distribution={distribution_name}.")
        if not set(distribution_params_bp.keys() - required_params).issubset(optional_params):
            raise Exception(f"Optional parameters accepted for misalignment distribution={distribution_name} are: {optional_params}.")
            
        distribution_params["scale"] = round(distribution_params_bp["scale_bp"] / bp_monomer)
        
        if "loc_bp" in distribution_params_bp:
            distribution_params["loc"] = round(distribution_params_bp["loc_bp"] / bp_monomer)
   
    if distribution_name == "exponnorm":
        
        required_params =  set(["scale_expon_bp", "scale_norm_bp",])
        optional_params =  set(["loc_bp",])
        
        if not required_params.issubset(distribution_params_bp.keys()):
            raise Exception(f"Values for {required_params} must be specified with misalignment distribution={distribution_name}.")
        if not set(distribution_params_bp.keys() - required_params).issubset(optional_params):
            raise Exception(f"Optional parameters accepted for misalignment distribution={distribution_name} are: {optional_params}.")
                
        distribution_params["K"] = round((distribution_params_bp["scale_expon_bp"] / bp_monomer) / (distribution_params_bp["scale_norm_bp"] / bp_monomer), 2)
        distribution_params["scale"] = round(distribution_params_bp["scale_norm_bp"] / bp_monomer)
        
        if "loc_bp" in distribution_params_bp:
            distribution_params["loc"] = round(distribution_params_bp["loc_bp"] / bp_monomer)
   
    if distribution_name == "laplace_asymmetric":

        required_params =  set(["scale_pos_bp", "scale_neg_bp",])
        optional_params =  set(["loc_bp",])
        
        if not required_params.issubset(distribution_params_bp.keys()):
            raise Exception(f"Values for {required_params} must be specified with misalignment distribution={distribution_name}.")
        if not set(distribution_params_bp.keys() - required_params).issubset(optional_params):
            raise Exception(f"Optional parameters accepted for misalignment distribution={distribution_name} are: {optional_params}.")
            
        distribution_params["kappa"] = round(sqrt((distribution_params_bp["scale_neg_bp"] / bp_monomer) / (distribution_params_bp["scale_pos_bp"] / bp_monomer)), 2)
        distribution_params["scale"] = round(sqrt((distribution_params_bp["scale_neg_bp"] / bp_monomer) * (distribution_params_bp["scale_pos_bp"] / bp_monomer)))
        
        if "loc_bp" in distribution_params_bp:
            distribution_params["loc"] = round(distribution_params_bp["loc_bp"] / bp_monomer)
        
    #if distribution_name == "linspace":
    #    accepted_params =   ["separation_bp",]
    
    #if distribution_name == "lognorm": to implement
   
    return distribution_params


def move_link_flavia(i, ups, downs):

    #print("i",i)
    #print("up", ups[i], "down", downs[i])
    
    misalignment = downs[i] - ups[i]
    #print("misalignment",misalignment)

    downs = np.delete(downs, i)
    ups = np.delete(ups, i)
    #print("ups", ups) 
    #print("downs", downs)
    
    # lists of nb of sites between two consecutive links excluding their positions 
    gaps_up = np.diff(ups) - 1
    gaps_down = np.diff(downs) - 1
    #print("gaps_up", gaps_up, len(gaps_up))
    #print("gaps_down", gaps_down, len(gaps_down))
        
    rng = np.random.default_rng()
    if misalignment >= 0:
        max_up_down_misalignments = np.array((downs[1:]-1) - (ups[:-1]+1), dtype=np.float64) # list of max misalignments in the up-down (up-site upstream of down-site) direction between two consecutive links excluding their positions 
        max_up_down_misalignments[max_up_down_misalignments < 0] = np.nan
        max_up_down_misalignments[(gaps_up==0) | (gaps_down==0)] = np.nan
        min_up_down_misalignments = np.array((downs[:-1]+1) - (ups[1:]-1), dtype=np.float64)  # list of min misalignments in the up-down (up-site upstream of down-site) direction between two consecutive links excluding their positions 
        min_up_down_misalignments[(min_up_down_misalignments < 0) & ~np.isnan(max_up_down_misalignments)] = 0
        min_up_down_misalignments[(min_up_down_misalignments < 0) & np.isnan(max_up_down_misalignments)] = np.nan
        min_up_down_misalignments[(gaps_up==0) | (gaps_down==0)] = np.nan
        #print("max_up_down_misalignments", max_up_down_misalignments, len(max_up_down_misalignments))
        #print("min_up_down_misalignments", min_up_down_misalignments, len(min_up_down_misalignments))
        allowed_gaps = np.logical_and.reduce([gaps_up>0, gaps_down>0, max_up_down_misalignments>=abs(misalignment), min_up_down_misalignments<=abs(misalignment)])
        #print("allowed_gaps", allowed_gaps)
        if len(np.where(allowed_gaps)[0])==0:
            #print("NO ALLOWED GAPS")
            return ups, downs
        
        #print("indices avail gaps ", np.where(allowed_gaps)[0])
        selected_gap = rng.choice(np.where(allowed_gaps)[0])
        #print("selected_gap id", selected_gap)
        allowed_ups = np.arange(max(ups[selected_gap]+1, (downs[selected_gap]+1)-abs(misalignment)), min((downs[selected_gap+1] - abs(misalignment)), ups[selected_gap+1]))
        #print(f"allowed_ups {allowed_ups}")
        selected_up = rng.choice(allowed_ups)
        selected_down = selected_up + abs(misalignment)
        #print(f"selected_up {selected_up} selected_up")
        #print(f"selected_down {selected_down}")
    else:
        max_down_up_misalignments = np.array((downs[:-1]+1) - (ups[1:]-1), dtype=np.float64)  # list of max misalignments in the down-up (down-site upstream of up-site) direction between two consecutive links excluding their positions
        max_down_up_misalignments[max_down_up_misalignments >= 0] = np.nan
        max_down_up_misalignments[(gaps_up==0) | (gaps_down==0)] = np.nan
        min_down_up_misalignments = np.array((downs[1:]-1) - (ups[:-1]+1), dtype=np.float64)  # list of min misalignments in the down-up (down-site upstream of up-site) direction between two consecutive links excluding their positions
        min_down_up_misalignments[(min_down_up_misalignments >= 0) & ~np.isnan(max_down_up_misalignments)] = -1
        min_down_up_misalignments[(min_down_up_misalignments >= 0) & np.isnan(max_down_up_misalignments)] = np.nan
        min_down_up_misalignments[(gaps_up==0) | (gaps_down==0)] = np.nan
        #print("max_down_up_misalignments", max_down_up_misalignments, len(max_down_up_misalignments))
        #print("min_down_up_misalignments", min_down_up_misalignments, len(min_down_up_misalignments))
        allowed_gaps = np.logical_and.reduce([gaps_up>0, gaps_down>0, abs(max_down_up_misalignments)>=abs(misalignment), abs(min_down_up_misalignments)<=abs(misalignment)])
        #print("allowed_gaps", allowed_gaps)
        if len(np.where(allowed_gaps)[0])==0:
            #print("NO ALLOWED GAPS")
            return ups, downs
        #print("indices avail gaps ", np.where(allowed_gaps)[0])
        selected_gap = rng.choice(np.where(allowed_gaps)[0])
        #print("selected_gap id", selected_gap)
        allowed_downs = np.arange(max(downs[selected_gap]+1, (ups[selected_gap]+1)-abs(misalignment)), min((ups[selected_gap+1] - abs(misalignment)), downs[selected_gap+1]))
        #print(f"allowed_downs {allowed_downs}")
        selected_down = rng.choice(allowed_downs)
        selected_up = selected_down + abs(misalignment)
        #print(f"selected_down {selected_down}")
        #print(f"selected_up {selected_up}")

    new_i_ups = bisect.bisect_right(ups, selected_up)
    new_i_downs = bisect.bisect_right(downs, selected_down)
    #print(f"new_i_ups {new_i_ups}")
    #print(f"new_i_downs {new_i_downs}")
    assert(new_i_ups == new_i_downs)
    new_i = new_i_ups
    ups = np.insert(ups, new_i, selected_up)
    downs = np.insert(downs, new_i, selected_down)
    
    return ups, downs


def mcmc_frozen_misalignments_flavia(chain_length, nb_links, ccohesin_bonds_distribution, closed_extremities=False, store_all=False):
    
    rng = np.random.default_rng()

    misalignments = []
    
    '''
    for distribution_name, values in ccohesin_bonds_distribution.items():

        # This can give some problems when the decimal of all populations are exactly .5 because they will all round up to one link more. 
        # But with a large number of links it should be negligible
        nb_link_population = round(values["population_frac"] * nb_links)
        if (nb_link_population==0):
            raise Exception(f'Too few links ({nb_links}) to have at least one link (>{values["population_frac"]*100}%) of "{distribution_name}".')

        misalignments.extend(getattr(scipy.stats, distribution_name)(**values["params"]).rvs(nb_link_population).round().astype(int))
    '''
    for distribution_id,distribution in ccohesin_bonds_distribution.items():

        # This can give some problems when the decimal of all populations are exactly .5 because they will all round up to one link more. 
        # But with a large number of links it should be negligible
        nb_link_population = round(distribution["population_frac"] * nb_links)
        if (nb_link_population==0):
            raise Exception(f'Too few links ({nb_links}) to have at least one link (>{distribution["population_frac"]*100}%) of {distribution["name"]}.')

        misalignments.extend(getattr(scipy.stats, distribution["name"])(**distribution["params"]).rvs(nb_link_population).round().astype(int))
    

    if len(misalignments) == nb_links+1: # this could be a problem of rounding, so we just randomly delete one link
        misalignments.pop(rng.integers(len(misalignments)))
    elif len(misalignments) != nb_links:
        raise Exception(f"Number of total links={nb_links} different from number of extracted misalignments={misalignments}")
        
    misalignments = np.sort(np.array(misalignments))
    
    ups = np.arange(nb_links,dtype=int)
    downs = ups + misalignments
    
    shift = min(ups.min(), downs.min()) - 1
    ups -= shift 
    downs -= shift 
    
    if downs.max() >= chain_length:
        raise ValueError("Misalignments are too large")

    ups = np.r_[0, ups, chain_length-1].astype(int)
    downs = np.r_[0, downs, chain_length-1].astype(int)
    
    idx_min = 0
    idx_max = len(ups) - 1
    
    if not closed_extremities:
        # don't store the positions of first and last links static at the extremities
        idx_min += 1
        idx_max -= 1
    
    if store_all:
        positions = [np.column_stack((ups[idx_min:idx_max+1], downs[idx_min:idx_max+1])),]
    
    for _ in range(nb_links*10):
                
        i = rng.integers(low=1, high=len(ups)-1) # don't move first and last static links at the extremities
        ups, downs = move_link_flavia(i, ups, downs)
        
        if store_all:
            positions.append(np.column_stack((ups[idx_min:idx_max+1], downs[idx_min:idx_max+1])))
            
    if not store_all:
        positions = np.column_stack((ups[idx_min:idx_max+1], downs[idx_min:idx_max+1]))
        
    return positions


def mc_cohesin_move(
    cohesin_bonds, 
    particles_position, 
    chain_length,
    sliding_links_configuration,
    update_odd = None, 
    cohesin_bond_k=0.3, 
    cohesin_bond_length=1,
    max_step=50,
    mask_updateable=None,
):
    
    n_cohesins = cohesin_bonds.shape[0]

    rng = np.random.default_rng()
    
    cohesin_steps = rng.integers(max_step*2+1, size=n_cohesins*2).reshape(n_cohesins,2)-max_step
    new_cohesin_bonds = cohesin_bonds+cohesin_steps

    if update_odd is None:
        odd_even_mask = np.full((n_cohesins, 2), True)
    else:
        odd_even_mask = (np.arange(0,n_cohesins)+1)%2 == update_odd
        odd_even_mask = np.vstack([odd_even_mask, odd_even_mask]).T
    
    constrained_left = new_cohesin_bonds[1:]<=cohesin_bonds[:-1]
    constrained_left = np.vstack([
        [new_cohesin_bonds[0,0]<0, new_cohesin_bonds[0,1]<chain_length],
        constrained_left])

    constrained_right = new_cohesin_bonds[:-1]>=cohesin_bonds[1:]
    constrained_right = np.vstack([
        constrained_right,
        [new_cohesin_bonds[-1,0]>=chain_length, new_cohesin_bonds[-1,1]>=2*chain_length]])

    step_is_possible = odd_even_mask & (~constrained_left) & (~constrained_right) & (sliding_links_configuration==1)
    
    if mask_updateable is not None:
        step_is_possible &= mask_updateable

    new_cohesin_bonds[~step_is_possible] = cohesin_bonds[~step_is_possible]

    old_dists = get_bond_lengths(cohesin_bonds, particles_position)
    new_dists = get_bond_lengths(new_cohesin_bonds, particles_position)

    old_energies = cohesin_bond_k / 2 * (old_dists - cohesin_bond_length) ** 2
    new_energies = cohesin_bond_k / 2 * (new_dists - cohesin_bond_length) ** 2

    accept_step = (new_energies <= old_energies) | (rng.random(size=n_cohesins) <= np.exp(-new_energies+old_energies))

    new_cohesin_bonds[~accept_step] = cohesin_bonds[~accept_step]

    return new_cohesin_bonds


def get_bond_lengths(bonds, particles_position):
    bond_coords = np.diff(particles_position[bonds],axis=1)[:,0]
    bond_lens = np.linalg.norm(bond_coords, axis=1)
    return bond_lens


def relocate_anchors(
    ccohesins_bonds,
    sliding_links_configuration,
    ccohesin_anchor_diffusion_distribution,
    chain_length,
):
    #print(ccohesins_bonds)
    
    anchors_order_maintained = False
    out_of_boundaries = True
    enough_space_between_anchors = False
    shifts_incompatible_with_links_distribution = True

    nb_anchors = [np.count_nonzero(sliding_links_configuration[:,chain] == 0) for chain in range(sliding_links_configuration.shape[1])]

    distribution_name = list(ccohesin_anchor_diffusion_distribution.keys())[0]
    values = ccohesin_anchor_diffusion_distribution[distribution_name]

    stop = 0
    anchors_not_ordered = 0
    links_out_of_boundaries = 0
    nb_shifts_incompatible_with_links_distribution = 0

    rng = np.random.default_rng()

    while (out_of_boundaries) or (not anchors_order_maintained) or (not enough_space_between_anchors) or (shifts_incompatible_with_links_distribution):

        anchors_order_maintained = True
        out_of_boundaries = False
        enough_space_between_anchors = True
        shifts_incompatible_with_links_distribution = False

        ccohesins_bonds_anchors_shifted = copy.deepcopy(ccohesins_bonds)
        #ccohesins_bonds_anchors_shifted[sliding_links_configuration == 0] += shifts
        #print(ccohesins_bonds_anchors_shifted)
        #print(sliding_links_configuration)

        for chain in range(ccohesins_bonds.shape[1]):

            shifts = getattr(scipy.stats, distribution_name)(**values["params"]).rvs(nb_anchors[chain]).round().astype(int)
            #print("shifts", shifts)

            anchors_already_moved = []

            #print("chain ", chain)

            anchor_ids = np.where(sliding_links_configuration[:,chain]==0)[0]
            #print("anchor_ids", anchor_ids)

            while (len(shifts)>0) and (not shifts_incompatible_with_links_distribution):

                #print("already moved", anchors_already_moved)

                distance_between_anchors = np.concatenate([np.array([ccohesins_bonds_anchors_shifted[anchor_ids[0],chain]]), np.diff(ccohesins_bonds_anchors_shifted[anchor_ids,chain]), np.array([chain_length - 1 - ccohesins_bonds_anchors_shifted[anchor_ids[-1],chain]])])
                #print("distance_between_anchors", distance_between_anchors)
                left_right_distance_between_anchors = distance_between_anchors[0:-1] + distance_between_anchors[1:]
                #print("sum distance_between_anchors", left_right_distance_between_anchors)
                left_right_distance_between_anchors_argsort = np.argsort(left_right_distance_between_anchors)
                #print("argsort distance_between_anchors", left_right_distance_between_anchors_argsort)
                left_right_distance_between_anchors_argsort = np.delete(left_right_distance_between_anchors_argsort, [ np.where(left_right_distance_between_anchors_argsort == j)[0][0] for j in [np.where(anchor_ids == i)[0][0] for i in anchors_already_moved] ])
                #print("argsort distance_between_anchors without already moved anchors", left_right_distance_between_anchors_argsort)

                #for left_right_distance_between_anchors_arg in left_right_distance_between_anchors_argsort:
                left_right_distance_between_anchors_arg = left_right_distance_between_anchors_argsort[0]
                #print("left_right_distance_between_anchors_arg", left_right_distance_between_anchors_arg)
                #print("left_right_distance", left_right_distance_between_anchors[left_right_distance_between_anchors_arg])
                anchor_id = anchor_ids[left_right_distance_between_anchors_arg]
                #print("anchor_id", anchor_id)
                left_dist = -distance_between_anchors[left_right_distance_between_anchors_arg]
                right_dist = distance_between_anchors[left_right_distance_between_anchors_arg+1]
                #print("left dist",left_dist, "  right dist",right_dist)
                left_not_anchored_link_id = anchor_ids[left_right_distance_between_anchors_arg-1]+1 if (anchor_id != anchor_ids[0]) else 0
                right_not_anchored_link_id = anchor_ids[left_right_distance_between_anchors_arg+1]-1 if (anchor_id != anchor_ids[-1]) else (ccohesins_bonds.shape[0] - 1)
                left_dist_excl_not_anchored_links = -distance_between_anchors[left_right_distance_between_anchors_arg]+(anchor_id-left_not_anchored_link_id)
                right_dist_excl_not_anchored_links = distance_between_anchors[left_right_distance_between_anchors_arg+1]-(right_not_anchored_link_id - anchor_id)
                #print("left dist without other links",left_dist_excl_not_anchored_links, "  right dist without other links",right_dist_excl_not_anchored_links)
                possible_shifts = shifts[(shifts>left_dist_excl_not_anchored_links) & (shifts<right_dist_excl_not_anchored_links)]
                #print("possible_shifts", possible_shifts)
                if len(possible_shifts) == 0:
                    shifts_incompatible_with_links_distribution = True
                    nb_shifts_incompatible_with_links_distribution += 1
                    break
                chosen_shift = rng.choice(possible_shifts)
                #print(chosen_shift)
                ccohesins_bonds_anchors_shifted[anchor_id,chain] += chosen_shift
                shifts = np.delete(shifts, np.where(shifts == chosen_shift)[0][0])
                #print("remaining shifts", shifts)
                anchors_already_moved.append(anchor_id)

                #print("\n")

            if shifts_incompatible_with_links_distribution:
                #print("-------------")
                #print("shifts_incompatible_with_links_distribution", shifts_incompatible_with_links_distribution)
                break


            #print(ccohesins_bonds_anchors_shifted[:,chain][sliding_links_configuration[:,chain] == 0])
            if not (np.all(np.diff(ccohesins_bonds_anchors_shifted[:,chain][sliding_links_configuration[:,chain] == 0]) > 0)):
                #print(f"Anchor order not maintained on chain {chain}")
                anchors_not_ordered += 1
                anchors_order_maintained = False
                #print(np.where(np.diff(ccohesins_bonds_anchors_shifted[:,chain][sliding_links_configuration[:,chain] == 0]) <= 0)[0])
                #for i in np.where(np.diff(ccohesins_bonds_anchors_shifted[:,chain][sliding_links_configuration[:,chain] == 0]) <= 0)[0]:
                #    print(ccohesins_bonds_anchors_shifted[sliding_links_configuration[:,chain] == 0][i-2:i+3])
                #    print(sliding_links_configuration[sliding_links_configuration[:,chain] == 0][i-2:i+3])
                #print('anchors_not_ordered',anchors_not_ordered)
                #print('links_out_of_boundaries',links_out_of_boundaries)
                #raise ValueError('too many tries')
            #else:
            #    anchors_order_maintained = True
            if (np.any(ccohesins_bonds_anchors_shifted[:,chain] < 0)) or (np.any(ccohesins_bonds_anchors_shifted[:,chain] > chain_length-1)):
                #print(f"Links out of boundaries on chain {chain}")
                out_of_boundaries = True
                links_out_of_boundaries += 1
            #else:
            #    out_of_boundaries = False

        if shifts_incompatible_with_links_distribution:
            continue        

        if (out_of_boundaries) or (not anchors_order_maintained):

            stop +=1
            if stop >1000000:
                print(np.where(np.diff(ccohesins_bonds_shifted_all[:,chain]) <= 0)[0])
                for i in np.where(np.diff(ccohesins_bonds_shifted_all[:,chain]) <= 0)[0]:
                    print(ccohesins_bonds_shifted_all[i-2:i+3])
                    print(sliding_links_configuration[i-2:i+3])
                print('anchors_not_ordered',anchors_not_ordered)
                print('links_out_of_boundaries',links_out_of_boundaries)
                raise ValueError('too many tries')

            continue

        print('anchors_not_ordered',anchors_not_ordered)
        print('links_out_of_boundaries',links_out_of_boundaries)
        print('nb_shifts_incompatible_with_links_distribution', nb_shifts_incompatible_with_links_distribution)
        print("***********************")
        ccohesins_bonds_shifted_all = copy.deepcopy(ccohesins_bonds_anchors_shifted)
        #print(ccohesins_bonds_shifted_all)

        for chain in range(ccohesins_bonds_anchors_shifted.shape[1]):
            #print(ccohesins_bonds_anchors_shifted[:,chain])
            #print(ccohesins_bonds_anchors_shifted[sliding_links_configuration[:,chain]==0,chain])
            anchor_ids = np.where(sliding_links_configuration[:,chain]==0)[0]
            #print(anchor_ids)
            prev_anchor_id = -1
            for anchor_id,anchor_position,previous_anchor_id,next_anchor_id in zip(anchor_ids, ccohesins_bonds_anchors_shifted[anchor_ids,chain], np.concatenate([[-1], anchor_ids[:-1]]), np.concatenate([anchor_ids[1:], [None]])):
                #print(anchor_id,anchor_position,previous_anchor_id,next_anchor_id, ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain], np.any(ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain] >= anchor_position), ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain][ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain] >= anchor_position], len(ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain][ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain] >= anchor_position]) )
                if np.any(ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain] >= anchor_position):
                    for i in range(1,len(ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain][ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain] >= anchor_position])+1):
                        #print("prev sites to reorder", ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain][ccohesins_bonds_anchors_shifted[previous_anchor_id+1:anchor_id,chain] >= anchor_position])
                        ccohesins_bonds_shifted_all[anchor_id-i,chain] = anchor_position - i
                elif np.any(ccohesins_bonds_anchors_shifted[anchor_id+1:next_anchor_id,chain] <= anchor_position):
                    for i in range(1,len(ccohesins_bonds_anchors_shifted[anchor_id+1:next_anchor_id,chain][ccohesins_bonds_anchors_shifted[anchor_id+1:next_anchor_id,chain] <= anchor_position])+1):
                        #print("next sites to reorder", ccohesins_bonds_anchors_shifted[anchor_id+1:next_anchor_id,chain][ccohesins_bonds_anchors_shifted[anchor_id+1:next_anchor_id,chain] <= anchor_position])
                        ccohesins_bonds_shifted_all[anchor_id+i,chain] = anchor_position + i


            if (np.any(ccohesins_bonds_shifted_all[:,chain] < 0)) or (np.any(ccohesins_bonds_shifted_all[:,chain] >= chain_length)):
                print(f"Links out of boundaries on chain {chain}")
                out_of_boundaries = True
            if np.any(np.diff(ccohesins_bonds_shifted_all[:,chain]) <= 0):
                print(f"Not enough space between anchors on chain {chain}")
                enough_space_between_anchors = False

        print('\n')

        stop +=1
        if stop >1000000:
            print(np.where(np.diff(ccohesins_bonds_shifted_all[:,chain]) <= 0)[0])
            for i in np.where(np.diff(ccohesins_bonds_shifted_all[:,chain]) <= 0)[0]:
                print(ccohesins_bonds_shifted_all[i-2:i+3])
                print(sliding_links_configuration[i-2:i+3])
            break

    ccohesins_bonds[:] = ccohesins_bonds_shifted_all
    #print(ccohesins_bonds_anchors_shifted)
    #print(ccohesins_bonds_shifted_all)
    #print(ccohesins_bonds)
    #print(np.all(ccohesins_bonds==ccohesins_bonds_shifted_all))
    
    #return ccohesins_bonds_shifted_all



##############################################################################################################


############# FUNCTIONS TO KEEP HERE #######################

def get_polymer_bonds(lo, hi, ring=False):
    
    bonds = np.vstack([np.arange(lo, hi-1), np.arange(lo+1, hi)]).T
    
    return bonds


def create_outdirname(parameters):
    
    if parameters["init_config"] is None:
    
        outdirname = f'model-{parameters["ccohesin_motility"]}'

        if parameters["ccohesin_motility"] == "static":

            outdirname = outdirname + f'_freqCcohesinBp-{parameters["freq_ccohesin_bp"]}'

            for distribution_name, distribution_params in parameters['ccohesin_misalignment_distribution_bp'].items():

                distribution_name = distribution_name.replace('_', '')
                outdirname = outdirname+f"_ccohesinMisalignmentDistributionBp-{distribution_name}"

                for param,value in distribution_params["params"].items():

                    param = re.sub(r'(_[a-z])', lambda pat: pat.group(1).upper().strip("_"), param)
                    outdirname = outdirname+"_"+"-".join([param, str(value)])

                outdirname = outdirname+"_frac-"+str(distribution_params["population_frac"])

        elif parameters["ccohesin_motility"] == "dynamic":

            outdirname = outdirname + f'_ccohesinNbFixedSides-{parameters["ccohesin_nb_fixed_sides"]}'

            if parameters["ccohesin_fixed_sides_conf"] is not None:
                outdirname = outdirname + f'_ccohesinFixedSidesConf-{parameters["ccohesin_fixed_sides_conf"]}'

            if parameters["ccohesin_distribution"] is not None:
                outdirname = outdirname + f'_ccohesinDistribution-{parameters["ccohesin_distribution"]}'
                
            if parameters["ccohesin_misalignment_distribution_bp"] is not None: 
                for distribution_name, distribution_params in parameters['ccohesin_misalignment_distribution_bp'].items():

                    distribution_name = distribution_name.replace('_', '')
                    outdirname = outdirname+f"_ccohesinMisalignmentDistributionBp-{distribution_name}"

                    for param,value in distribution_params["params"].items():

                        param = re.sub(r'(_[a-z])', lambda pat: pat.group(1).upper().strip("_"), param)
                        outdirname = outdirname+"_"+"-".join([param, str(value)])

                    outdirname = outdirname+"_frac-"+str(distribution_params["population_frac"])

            if parameters["freq_ccohesin_bp"] is not None:
                outdirname = outdirname + f'_freqCcohesinBp-{parameters["freq_ccohesin_bp"]}'
            #if parameters["intrapair_freq_ccohesin_bp"] is not None:
            #    outdirname = outdirname + f'_intrapairFreqCcohesinBp-{parameters["intrapair_freq_ccohesin_bp"]}'
            #if parameters["interpair_freq_ccohesin_bp"] is not None:
            #    outdirname = outdirname + f'_interpairFreqCcohesinBp-{parameters["interpair_freq_ccohesin_bp"]}'
            if parameters["bp_dist_ccohesin_anchors_01"] is not None:
                outdirname = outdirname + f'_bpDistCcohesinAnchors01-{parameters["bp_dist_ccohesin_anchors_01"]}'
            if parameters["bp_dist_ccohesin_anchors_10"] is not None:
                outdirname = outdirname + f'_bpDistCcohesinAnchors10-{parameters["bp_dist_ccohesin_anchors_10"]}'
            if "nb_ccohesin_same_anchor" in parameters.keys():
                outdirname = outdirname + f'_nbCcohesinSameAnchor-{parameters["nb_ccohesin_same_anchor"]}'
            if "add_boundaries" in parameters.keys():
                outdirname = outdirname + f'_addBoundaries-{parameters["add_boundaries"]}'
            if parameters["nb_mcmc_moves_per_block"] is not None:
                outdirname = outdirname + f'_nbMcmcMovesPerBlock-{parameters["nb_mcmc_moves_per_block"]}'
            if parameters["max_mcmc_step"] is not None:
                outdirname = outdirname + f'_maxMcmcStep-{parameters["max_mcmc_step"]}'

            if ("add_loops" in parameters) and (parameters["add_loops"]):
                #outdirname = outdirname + f'_addLoops-{parameters["add_loops"]}'
                outdirname = outdirname + f'_loopLenBp-{parameters["avg_loop_length_bp"]}'
                outdirname = outdirname + f'_loopCoverage-{parameters["max_percentage_loop_coverage"]}'

            outdirname = outdirname + f'_nbBlocks-{parameters["nb_blocks"]}_blockSize-{parameters["block_size"]}'
        
    else:
        init_dirname = pathlib.Path(parameters["init_config"]).parent.parent.name
        init_dirname_split = init_dirname.split('_')
        outdirname = '_'.join([substr for substr in init_dirname_split if not substr.startswith(("nbMcmcMovesPerBlock", "maxMcmcStep", "nbBlocks", "blockSize"))])
        outdirname = f'{outdirname}_lastFrame-copied'

        if ("add_loops" in parameters) and (parameters["add_loops"]):
            #outdirname = outdirname + f'_addLoops-{parameters["add_loops"]}'
            outdirname = outdirname + f'_loopLenBp-{parameters["avg_loop_length_bp"]}'
            outdirname = outdirname + f'_loopCoverage-{parameters["max_percentage_loop_coverage"]}'

            outdirname = outdirname + f'_nbBlocks-{parameters["nb_blocks"]}_blockSize-{parameters["block_size"]}'
        
    return outdirname


def check_args(
    bp_monomer,
    ccohesin_motility,
    ccohesin_nb_fixed_sides,
    ccohesin_fixed_sides_conf,
    freq_ccohesin_bp,
    bp_dist_ccohesin_anchors_01,
    bp_dist_ccohesin_anchors_10,
    ccohesin_distribution,
    ccohesin_misalignment_distribution_bp,
    nb_mcmc_moves_per_block,
    max_mcmc_step,
    chain_first_anchor,
):
    
    if (freq_ccohesin_bp is not None) and (freq_ccohesin_bp < bp_monomer):
        raise TypeError(f'freq_ccohesin_bp should be greater or equal than bp_monomer')
    if (bp_dist_ccohesin_anchors_01 is not None) and (bp_dist_ccohesin_anchors_01 < bp_monomer):
        raise TypeError(f'bp_dist_ccohesin_anchors_01 should be greater or equal than bp_monomer')
    if (bp_dist_ccohesin_anchors_10 is not None) and (bp_dist_ccohesin_anchors_10 < bp_monomer):
        raise TypeError(f'bp_dist_ccohesin_anchors_10 should be greater or equal than bp_monomer')
            
    if ccohesin_motility is None:
        raise ValueError(f'Argument --ccohesin_motility cannot be None. Choices are "static"|"dynamic"')
        
    elif ccohesin_motility == "static":
        if ccohesin_nb_fixed_sides != 2:
            raise ValueError(f'When ccohesin_motility=="static", ccohesin_nb_fixed_sides must be set to 2')
        if ccohesin_fixed_sides_conf is not None:
            raise TypeError(f'When ccohesin_motility=="static", ccohesin_fixed_sides_conf must be None')
        if bp_dist_ccohesin_anchors_01 is not None:
            raise TypeError(f'When ccohesin_motility=="static", bp_dist_ccohesin_anchors_01 must be None')
        if bp_dist_ccohesin_anchors_10 is not None:
            raise TypeError(f'When ccohesin_motility=="static", bp_dist_ccohesin_anchors_10 must be None')
        if ccohesin_distribution is not None:
            raise TypeError(f'When ccohesin_motility=="static", ccohesin_distribution must be None')
        if nb_mcmc_moves_per_block is not None:
            raise TypeError(f'When ccohesin_motility=="static", nb_mcmc_moves_per_block must be None')
        if max_mcmc_step is not None:
            raise TypeError(f'When ccohesin_motility=="static", max_mcmc_step must be None')
        if freq_ccohesin_bp is None:
            raise TypeError(f'When ccohesin_motility=="static", freq_ccohesin_bp must not be None. ')
        if ccohesin_misalignment_distribution_bp is None:
            raise TypeError(f'When ccohesin_motility=="static", ccohesin_misalignment_distribution_bp must not be None. Use --help for more details.')
        if chain_first_anchor is not None:
            raise TypeError(f'When ccohesin_motility=="static", <chain_first_anchor> must be None')

    elif ccohesin_motility == "dynamic":
        if (ccohesin_nb_fixed_sides != 0) and (ccohesin_nb_fixed_sides != 1):
            raise ValueError(f'When ccohesin_motility=="dynamic", ccohesin_nb_fixed_sides must be set to 0|1')
        if (ccohesin_nb_fixed_sides == 0) and (ccohesin_fixed_sides_conf is not None):
            raise ValueError(f'When ccohesin_motility=="dynamic" and ccohesin_nb_fixed_sides==0, ccohesin_fixed_sides_conf must be None')
        if (ccohesin_nb_fixed_sides == 1) and (ccohesin_fixed_sides_conf is None):
            raise ValueError(f'When ccohesin_motility=="dynamic" and ccohesin_nb_fixed_sides==1, ccohesin_fixed_sides_conf must be set to "random"|"opposite"|"same"')
        if (
            (freq_ccohesin_bp is None) and ((bp_dist_ccohesin_anchors_01 is None) or (bp_dist_ccohesin_anchors_10 is None))
            or (freq_ccohesin_bp is not None) and ((bp_dist_ccohesin_anchors_01 is not None) or (bp_dist_ccohesin_anchors_10 is not None))
        ):
            raise ValueError(f'When ccohesin_motility=="dynamic", either freq_ccohesin_bp or (bp_dist_ccohesin_anchors_01 & bp_dist_ccohesin_anchors_10) must be an integer, not None')       
        #if ccohesin_distribution is None:
        #    raise ValueError(f'When ccohesin_motility=="dynamic", ccohesin_distribution must be "expon"|"linspace"')
        #if ccohesin_misalignment_distribution_bp is not None:
        #    raise TypeError(f'When ccohesin_motility=="dynamic", ccohesin_misalignment_distribution_bp must be None')
        #if nb_mcmc_moves_per_block is None:
        #    raise TypeError(f'When ccohesin_motility=="dynamic", nb_mcmc_moves_per_block must be an integer, not None')
        #if max_mcmc_step is None:
        #    raise TypeError(f'When ccohesin_motility=="dynamic", max_mcmc_step must be an integer, not None')
        if (chain_first_anchor is not None) and ((bp_dist_ccohesin_anchors_01 is not None) and (bp_dist_ccohesin_anchors_10 is not None)):
            raise ValueError(f'<chain_first_anchor> must be None, it will be automatically defined based on the anchor distances.')
        if (chain_first_anchor != 0) and (chain_first_anchor != 1) and (chain_first_anchor is not None) and ((bp_dist_ccohesin_anchors_01 is None) and (bp_dist_ccohesin_anchors_10 is None)):
            raise ValueError(f'<chain_first_anchor> must be None|0|1')


def run_sim(outdir, 
            chain_length,
            ccohesin_motility,
            ccohesin_nb_fixed_sides,
            ccohesin_fixed_sides_conf,
            freq_ccohesin_bp,
            bp_dist_ccohesin_anchors_01,
            bp_dist_ccohesin_anchors_10,
            ccohesin_distribution,
            ccohesin_misalignment_distribution_bp,
            ccohesin_anchor_diffusion_distribution_bp,
            nb_mcmc_moves_per_block,
            max_mcmc_step,
            nb_blocks, 
            block_size=100000,
            bp_monomer=200, 
            A_dpd=10, 
            gamma_dpd=10,
            replicate_id=None, 
            init_config=None,
            overwrite=False,
            create_new_outdir=False,
            **kwargs,
           ):
    
    density = kwargs.get('density', 0.1)
    dt = kwargs.get('dt', 0.5)
    ccohesin_bond_k = kwargs.get('ccohesin_bond_k', 0.3)
    ccohesin_bond_length = kwargs.get('ccohesin_bond_length', 1)
    polymer_bond_k = kwargs.get('polymer_bond_k', 100)
    polymer_bond_length = kwargs.get('polymer_bond_length', 1)
    add_boundaries = kwargs.get('add_boundaries', False)
    nb_ccohesin_same_anchor = kwargs.get('nb_ccohesin_same_anchor', 1)
    chain_first_anchor = kwargs.get('chain_first_anchor', None)
    
    
    check_args(
        bp_monomer,
        ccohesin_motility,
        ccohesin_nb_fixed_sides,
        ccohesin_fixed_sides_conf,
        freq_ccohesin_bp,
        bp_dist_ccohesin_anchors_01,
        bp_dist_ccohesin_anchors_10,
        ccohesin_distribution,
        ccohesin_misalignment_distribution_bp,
        nb_mcmc_moves_per_block,
        max_mcmc_step,
        chain_first_anchor,
    )
    
    n_tot = chain_length * 2
    
    if freq_ccohesin_bp is not None:
        freq_ccohesin = round(freq_ccohesin_bp / bp_monomer)
        n_ccohesin = round(chain_length / freq_ccohesin)
    elif (bp_dist_ccohesin_anchors_01 is not None) and (bp_dist_ccohesin_anchors_10 is not None):
        dist_ccohesin_anchors_01 = round(bp_dist_ccohesin_anchors_01 / bp_monomer)
        dist_ccohesin_anchors_10 = round(bp_dist_ccohesin_anchors_10 / bp_monomer)
    
    #mitotic chromosomes have density of 1 nucleosome (diameter ~10nm)->10^3 / (14 * 14 * 14) nm^3; 
    #interphase chromsomes occupy 3-10x bigger volume
    #BOX_PER_PARTICLE = 2 * (3**(1/3))
    #BOX_PER_PARTICLE = 10**(1/3)
    #BOX_SIZE = (N**(1/3)) * BOX_PER_PARTICLE
    #DENSITY = round((N/(BOX_SIZE**3)),1)
    
    ################################################################################
    #####################    SETUP OUTPUT FOLDER AND FILES    ######################
    ################################################################################
    
    if (outdir is None) or (create_new_outdir):
        outdir = f'{str(outdir or ".")}/{create_outdirname(locals())}/{args.replicate_id if args.replicate_id is not None else ""}'
    
    outdir = pathlib.Path(outdir)
    
    '''
    sim_name_dict = dict(
        #N_TOT=N,
        #CHAINLEN=chain_length,
        #MONOMERBP=bp_monomer,
        FREQCCOHESBP=freq_ccohesin_bp,
        #BLOCKS=nb_blocks,
        #BLOCKSIZE=block_size,
        #BOX=BOX_PER_PARTICLE,
        #D=density,
        #DT=dt,
        #A=A,
        #GAMMA=gamma,
        #CCOHESLEN=ccohesin_bond_length,
        #CCOHESK=ccohesin_bond_k,
        #CHAINBONDLEN=polymer_bond_length,
        #CHAINBONDK=polymer_bond_k,
        #R=replicate,
    )
    
    sim_name = "_".join(f'{n}-{v}' for n, v in sim_name_dict.items())
    

    if replicate_id is not None:
        out_folder = out_root_folder / sim_name / replicate_id
    else:
        out_folder = out_root_folder / sim_name
    '''
    
    if outdir.exists():
        if not overwrite:
            raise Exception(f'The output folder {outdir} already exists! If you want to overwrite it, pass overwrite=True as command line argument.')
    else:
        outdir.mkdir(parents=True)
    
    
    logging.info(f'Output folder: {outdir}')

    out_init_file = f"{outdir}/initial_conformation.gsd"
    out_minimized_file =  f"{outdir}/energy_minimized_conformation.gsd"
    out_compressed_file =  f"{outdir}/compressed_conformation.gsd"
    out_preequilibration_traj_file =  f"{outdir}/preequilibration_traj.gsd"
    out_trajectory_file =  f"{outdir}/trajectory.gsd"
    
    logging.info(f'All simulation variables: {locals()}')
    
    #print(f'All simulation variables: {locals()}')
    
    ################################################################################
    #################    SET UP COHESIVE COHESINS CONFIGURATION    #################
    ################################################################################
    
    if ccohesin_motility == "static":
    
        ccohesin_misalignment_distribution = copy.deepcopy(ccohesin_misalignment_distribution_bp)

        #for distribution_name in ccohesin_misalignment_distribution_bp:
            #distribution_params_bp = ccohesin_misalignment_distribution_bp[distribution_name]["params"]
            #converted_distribution_params = convert_misalignment_distribution_params(distribution_name, distribution_params_bp, bp_monomer=bp_monomer)
            #ccohesin_misalignment_distribution[distribution_name]["params"] = converted_distribution_params
        for distribution_id,distribution in ccohesin_misalignment_distribution_bp.items():
            distribution_name = distribution['name']
            distribution_params_bp = distribution["params"]
            converted_distribution_params = convert_misalignment_distribution_params(distribution_name, distribution_params_bp, bp_monomer=bp_monomer)
            ccohesin_misalignment_distribution[distribution_id]["params"] = converted_distribution_params

        ccohesins_bonds = mcmc_frozen_misalignments_flavia(chain_length, n_ccohesin, ccohesin_misalignment_distribution)
        #ccohesins_bonds[:,1] += chain_length
        
    elif ccohesin_motility == "dynamic":
        
        if ccohesin_distribution == "linspace":
            
            ccohesins_bonds = np.linspace(0, chain_length-1, num=n_ccohesin, dtype='int')
            ccohesins_bonds = np.column_stack((ccohesins_bonds,ccohesins_bonds))
            #ccohesins_bonds[:,1] += chain_length
            
        elif ccohesin_distribution == "expon":

            if freq_ccohesin_bp is not None:
                ccohesins_bonds = exponentially_separated_links(chain_length, 1/freq_ccohesin, add_boundaries=add_boundaries)
            elif (bp_dist_ccohesin_anchors_01 is not None) and (bp_dist_ccohesin_anchors_10 is not None):
                if bp_dist_ccohesin_anchors_01 <= bp_dist_ccohesin_anchors_10:
                    intrapair_dist_ccohesin_anchors = dist_ccohesin_anchors_01
                    interpair_dist_ccohesin_anchors = dist_ccohesin_anchors_10
                    chain_first_anchor = 0
                else:
                    intrapair_dist_ccohesin_anchors = dist_ccohesin_anchors_10
                    interpair_dist_ccohesin_anchors = dist_ccohesin_anchors_01
                    chain_first_anchor = 1

                #ccohesins_bonds, _ = exponentially_separated_link_pairs(
                #    chain_length,
                #    frequency_intrapair_links = 1/intrapair_dist_ccohesin_anchors,
                #    frequency_interpair_links = 1/interpair_dist_ccohesin_anchors,
                #    min_separation=1,
                #    add_boundaries=add_boundaries,
                #)
                
                ccohesins_bonds, _ = exponentially_separated_semianchored_links(
                    chain_length,
                    frequency_intrapair_links = 1/intrapair_dist_ccohesin_anchors,
                    frequency_interpair_links = 1/interpair_dist_ccohesin_anchors,
                    nb_links_same_anchor=nb_ccohesin_same_anchor,
                    min_separation=1,
                    add_boundaries=add_boundaries,
                )

                #ccohesins_bonds[:,1] += chain_length
         
        '''  Uncomment for diffusion
        elif ccohesin_misalignment_distribution_bp is not None:
            
            ccohesin_misalignment_distribution = copy.deepcopy(ccohesin_misalignment_distribution_bp)

            for distribution_name in ccohesin_misalignment_distribution_bp:
                distribution_params_bp = ccohesin_misalignment_distribution_bp[distribution_name]["params"]
                converted_distribution_params = convert_misalignment_distribution_params(distribution_name, distribution_params_bp, bp_monomer=bp_monomer)
                ccohesin_misalignment_distribution[distribution_name]["params"] = converted_distribution_params

            ccohesins_bonds = mcmc_frozen_misalignments_flavia(chain_length, n_ccohesin, ccohesin_misalignment_distribution)
        '''
            

    sliding_links_configuration = set_sliding_links_configuration(
        ccohesins_bonds,
        ccohesin_nb_fixed_sides,
        ccohesin_fixed_sides_conf,
        chain_first_anchor=chain_first_anchor,
        set_boundaries=add_boundaries,
        nb_same_anchor=nb_ccohesin_same_anchor,
    )

    if ccohesin_anchor_diffusion_distribution_bp is not None:

        ccohesin_anchor_diffusion_distribution = copy.deepcopy(ccohesin_anchor_diffusion_distribution_bp)

        for distribution_name in ccohesin_anchor_diffusion_distribution_bp:
            distribution_params_bp = ccohesin_anchor_diffusion_distribution_bp[distribution_name]["params"]
            converted_distribution_params = convert_misalignment_distribution_params(distribution_name, distribution_params_bp, bp_monomer=bp_monomer)
            ccohesin_anchor_diffusion_distribution[distribution_name]["params"] = converted_distribution_params

        relocate_anchors(ccohesins_bonds, sliding_links_configuration, ccohesin_anchor_diffusion_distribution, chain_length)
        
    ccohesins_bonds[:,1] += chain_length
        
    logging.info(f'\n {ccohesins_bonds[:50]}')
    logging.info(f'\n {ccohesins_bonds.shape}')
    logging.info(f'\n {sliding_links_configuration[:50]}')
    logging.info(f'\n {sliding_links_configuration.shape}')
    
    
    ################################################################################
    #######################    INITIALIZE THE SIMULATION    ########################
    ################################################################################
    
    gpu = hoomd.device.GPU()
    sim = hoomd.Simulation(device=gpu, seed=1)
    logger = hoomd.logging.Logger()
    
    ################################################################################
    #########################    INITIALIZE THE SYSTEM    ##########################
    ################################################################################
    
    # get the start datetime to setup initial configuration
    st_init_config = datetime.datetime.now()
    
    # Set the initial configuration of the system and the topology
    #init_snapshot = gsd.hoomd.Snapshot() #Deprecated
    init_snapshot = gsd.hoomd.Frame()

    init_snapshot.particles.N = n_tot
    
    # Hoomd has a built-in concept of particle types! 
    # The types are integer-coded, but there is also a table (snapshot.particles.types) that sets a string identifier for each type.
    init_snapshot.particles.types = ['chromatin']

    typeids = np.zeros(n_tot, dtype=np.int32)
    init_snapshot.particles.typeid = typeids
    
    #Set bonds
    polymer_bonds = np.r_[
        get_polymer_bonds(0, chain_length),
        get_polymer_bonds(chain_length, 2*chain_length)]   #get_polymer_bonds(L_CHAIN+1, 2*L_CHAIN)] from example_hoomd-moving-cohesins doesn't take into account first bond of sister 2

    all_bonds = np.r_[polymer_bonds, ccohesins_bonds]
    
    logger[('sliding_links_configuration',)] = (lambda: np.vstack([np.full(polymer_bonds.shape, np.nan),sliding_links_configuration]), 'bond')

    init_positions = cohesed_sisters_starting_configuration_new(chain_length, ccohesins_bonds)
    #init_positions = cohesed_sisters_starting_configuration(chain_length, ccohesins_bonds)
    init_positions -= np.mean(init_positions, axis=0)

    max_dist_from_center = ceil(max(np.linalg.norm(init_positions - init_positions.mean(axis=0), axis=1))) + 1
    #print("max dist center", max_dist_from_center)

    init_snapshot.configuration.box = [max_dist_from_center*2, max_dist_from_center*2, max_dist_from_center*2, 0, 0, 0]

    init_snapshot.particles.position = init_positions

    init_snapshot.bonds.N = all_bonds.shape[0] 
    init_snapshot.bonds.group = all_bonds
    init_snapshot.bonds.types = ['polymer', 'ccohesin']
    init_snapshot.bonds.typeid = np.r_[
        np.zeros(polymer_bonds.shape[0]),
        np.ones(ccohesins_bonds.shape[0])]
    
    init_snapshot.bonds.validate()
    
    # The initial configuration is saved into a file to be loaded immediately.
    with gsd.hoomd.open(name=out_init_file, mode='wb') as f:
        f.append(init_snapshot)

    # get the end datetime to setup initial configuration
    et_init_config = datetime.datetime.now()

    # get execution time to setup initial configuration
    elapsed_time_init_config = et_init_config - st_init_config
    logging.info(f'Execution time to setup initial conformation: {elapsed_time_init_config} seconds')

    
    ################################################################################
    ############################    SETUP SIMULATION    ############################
    ################################################################################

    # get the start datetime of loading initial conformation and setting sim
    st_setup = datetime.datetime.now()

    # set coordinates
    sim.create_state_from_gsd(filename=out_init_file)

    # set speeds
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)

    # creates bonded forces
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params['polymer'] = dict(k=polymer_bond_k, r0=polymer_bond_length)
    harmonic.params['ccohesin'] = dict(k=ccohesin_bond_k, r0=ccohesin_bond_length)

    # create non-bonded forces
    # non-bonded forces require a neighbour list, which maintains a list of interacting particle pairs
    # use Tree for big and/or sparse systems; use Cell for smaller or compacted systems
    nlist = hoomd.md.nlist.Tree(buffer=0.4, exclusions=('bond',))
    #nlist = hoomd.md.nlist.Stencil(cell_width=10.0, buffer=0.4)
    #nlist = hoomd.md.nlist.Cell(buffer=0.4, exclusions=('bond',))
    
    # DPD is implemented as a whole package of 3 forces: repulsion, friction between colliding particles and random force 
    dpd = hoomd.md.pair.DPD(nlist=nlist, kT=1.0, default_r_cut=1.0)
    # non-bonded forces must be configured for every pair of interacting particle types
    dpd.params.default = dict(A=A_dpd, gamma=gamma_dpd)
    
    #print("DPD LOGGABLES", dpd.loggables)
    #print(repr(dpd.params))
    
    # LJ
    ##sigma = 1.0
    ##epsilon = 1.0
    ##r_cut=2**(1./6.)*sigma
    #r = np.linspace(0.5, 1.5, 500)
    #V_lj = 4 * epsilon * ((sigma / r)**12 - (sigma / r)**6)
    ##lj0804 = hoomd.md.pair.LJ0804(nlist, default_r_cut=r_cut)
    ##lj0804.params[('chromatin','chromatin')] = {'sigma': sigma, 'epsilon': epsilon}

    thermodynamic_properties = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermodynamic_properties)
    
    particle_filter = hoomd.filter.Type(['chromatin'])
    
    # get the end datetime of loading initial conformation and setting sim
    et_setup = datetime.datetime.now()

    # get execution time of loading initial conformation and setting sim
    elapsed_time_setup = et_setup - st_setup
    logging.info(f'Execution time to load initial conformation and setup the simulation: {elapsed_time_setup} seconds')
    
    ################################################################################
    ################    MINIMIZE POTENTIAL ENERGY OF THE SYSTEM    #################
    ################################################################################
    
    # get the start datetime of minimization step
    st_minimization = datetime.datetime.now()
    
    # NVE equations of motion == newton equations + Verlet-Velocity algorithm 
    nve_fire = hoomd.md.methods.NVE(filter=particle_filter)
    
    fire = hoomd.md.minimize.FIRE(
        dt=0.05,
        force_tol=5e-2,
        angmom_tol=5e-2,
        energy_tol=5e-2,
        # alpha_start=0.999,
        forces=[harmonic, dpd], #lj0804, lj],
        methods=[nve_fire],
        )

    sim.operations.integrator = fire
    sim.run(0)
    
    logging.info(f'Initial thermodynamic properties of the system')
    logging.info(f'kin temp = {thermodynamic_properties.kinetic_temperature}, E_P/N = {thermodynamic_properties.potential_energy / n_tot}, E_P = {thermodynamic_properties.potential_energy}')

    gsd_minimized_writer = hoomd.write.GSD(
        filename=out_minimized_file,
        trigger=hoomd.trigger.Periodic(10),
        #trigger=hoomd.trigger.Periodic(BLOCK_SIZE),
        #dynamic=['momentum',],
        mode='wb')

    sim.operations.writers.append(gsd_minimized_writer)
    
    while not( (thermodynamic_properties.potential_energy / n_tot) < 1 ):
        logging.info('run FIRE')
        sim.run(10)
        logging.info(f'kin temp = {thermodynamic_properties.kinetic_temperature}, E_P/N = {thermodynamic_properties.potential_energy / n_tot}, E_P = {thermodynamic_properties.potential_energy}')
        
    for _ in range(len(fire.forces)):
        fire.forces.pop()

    sim.operations.writers.pop(0)

    # get the end datetime of minimization step
    et_minimization = datetime.datetime.now()

    # get execution time of the minimization step
    elapsed_time_minimization = et_minimization - st_minimization
    logging.info(f'Execution time to minimize energy of initial conformation: {elapsed_time_minimization} seconds')
    
    ################################################################################
    ############################    SET DPD INTEGRATOR   ###########################
    ################################################################################
    
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    
    
    nve_dpd_integrator = hoomd.md.methods.NVE(filter=particle_filter)

    # the integrator updates the positions and velocities of particles based on forces
    # note that we can get away with a very large timestep. All units in hoomd are normalized, so that speeds=1.0, particle diameters=1.0, etc.
    dpd_integrator = hoomd.md.Integrator(
        dt=dt, 
        methods=[nve_dpd_integrator], 
        forces=[harmonic, dpd])
    sim.operations.integrator = dpd_integrator
        
    '''
    langevin = hoomd.md.methods.Langevin(filter=particle_filter, kT=1.0)
    langevin.gamma.default = gamma
    #method = langevin
    langevin_integrator = hoomd.md.Integrator(
        dt=dt, 
        methods=[langevin], 
        forces=[harmonic, lj0804],
    )
    sim.operations.integrator = langevin_integrator 

    
    dpd
    brownian = hoomd.md.methods.Brownian(filter=particle_filter, kT=kT)
    brownian.gamma.default = gamma
    method = brownian
    '''

    ################################################################################
    ######################    COMPRESS THE SYSTEM IF NEEDED   ######################
    ################################################################################
    
    # Compress the system if the initial density is lower than the desired one
    initial_density = sim.state.N_particles / sim.state.box.volume
    final_density = density
    
    if initial_density < final_density:
        
        # get the start datetime of system compression
        st_compression = datetime.datetime.now()

        gsd_compressed_writer = hoomd.write.GSD(
            filename=out_compressed_file,
            trigger=hoomd.trigger.Periodic(1000),
            dynamic=['momentum',],
            mode='wb')

        sim.operations.writers.append(gsd_compressed_writer)

        #sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    
        ramp = hoomd.variant.Ramp(A=0, B=1, t_start=sim.timestep, t_ramp=10000)
        initial_box = sim.state.box
        logging.info(f"Density before compression = {initial_density}") 
        logging.info(f"Box size before compression = {initial_box.L}")

        final_box = hoomd.Box.from_box(initial_box)  # make a copy of initial_box
        final_box.volume = sim.state.N_particles / final_density
        box_resize_trigger = hoomd.trigger.Periodic(10)
        box_resize = hoomd.update.BoxResize(box1=initial_box,
                                            box2=final_box,
                                            variant=ramp,
                                            trigger=box_resize_trigger)
        sim.operations.updaters.append(box_resize)

        logging.info('Compressing...')

        for cur_step in range(0, 10000, 1000):
            logging.info(f'compression step={cur_step}')
            sim.run(1000)
            logging.info(f'kin temp = {thermodynamic_properties.kinetic_temperature}, E_P/N = {thermodynamic_properties.potential_energy / n_tot}')

        final_density = sim.state.N_particles / sim.state.box.volume
        final_box = sim.state.box
        logging.info(f"Density after compression = {final_density}") 
        logging.info(f"Box size after compression = {final_box.L}")

        sim.operations.updaters.remove(box_resize)
        sim.operations.writers.pop(0)

        # get the end datetime of system compression
        et_compression = datetime.datetime.now()

        # get execution time of system compression
        elapsed_time_compression = et_compression - st_compression
        logging.info(f'Execution time to compress the system: {elapsed_time_compression} seconds')
    else:
        logging.info(f'Compression not needed. Density {initial_density} already <= {density}.')
    
    ''' Uncomment for diffusion
    ################################################################################
    #########################    EQUILIBRATE THE SYSTEM 1   ########################
    ################################################################################
    
    # get the start datetime of all blocks
    st_preequilibration = datetime.datetime.now()
        
    nb_blocks_preequilibration = 10
    block_size_preequilibration = 1000000
    gsd_trajectory_writer = hoomd.write.GSD(
        filename=out_preequilibration_traj_file,
        trigger=hoomd.trigger.Periodic(block_size_preequilibration),
        dynamic=['momentum', 'topology',],
        mode='wb',
        logger=logger,
    )

    sim.operations.writers.append(gsd_trajectory_writer)
    
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    
    for cur_step in range(0, nb_blocks_preequilibration):
        logging.info(f'preequilibration block index={cur_step}')

        # get the start datetime of each block
        st_block_preequilibration = datetime.datetime.now()

        sim.run(block_size_preequilibration)

        # get the end datetime
        et_block_preequilibration = datetime.datetime.now()

        # get execution time
        elapsed_time_block_preequilibration = et_block_preequilibration - st_block_preequilibration
        logging.info(f'Execution time of preequilibration block {cur_step}: {elapsed_time_block_preequilibration} seconds')

        logging.info(f'kin temp = {thermodynamic_properties.kinetic_temperature}, E_P/N = {thermodynamic_properties.potential_energy / n_tot}')
    
    # get the end datetime
    et_preequilibration = datetime.datetime.now()

    # get execution time
    elapsed_time_preequilibration = et_preequilibration - st_preequilibration
    logging.info(f'Execution time of preequilibration: {elapsed_time_preequilibration} seconds')
    '''
    
    ################################################################################
    #########################    EQUILIBRATE THE SYSTEM 2   ########################
    ################################################################################
    
    # get the start datetime of all blocks
    st_all_blocks = datetime.datetime.now()
        
    gsd_trajectory_writer = hoomd.write.GSD(
        filename=out_trajectory_file,
        trigger=hoomd.trigger.Periodic(int((block_size*nb_blocks)/100)),
        #trigger=hoomd.trigger.Periodic(block_size),
        dynamic=['momentum', 'topology',],
        mode='wb',
        logger=logger,
    )

    sim.operations.writers.append(gsd_trajectory_writer)
    
    sim.state.thermalize_particle_momenta(filter=hoomd.filter.All(), kT=1.0)
    
    
    if (ccohesin_motility == "dynamic"):
        
        if max_mcmc_step is None:
            dist_alternate_ccohesins_chain0 = np.diff(ccohesins_bonds[np.where(sliding_links_configuration[:,0]==0)[0],0])
            dist_alternate_ccohesins_chain1 = np.diff(ccohesins_bonds[np.where(sliding_links_configuration[:,1]==0)[0],1]-chain_length)
            max_mcmc_step = np.max(np.concatenate([dist_alternate_ccohesins_chain0, dist_alternate_ccohesins_chain1]))
            logging.info(f'Setting <max_mcmc_step> to the max distance between two consecutive anchors on the same chain => max_mcmc_step={max_mcmc_step}.')
        if nb_mcmc_moves_per_block is None:
            nb_mcmc_moves_per_block = int(max_mcmc_step*0.2)
            logging.info(f'Setting <nb_mcmc_moves_per_block> to 20% of <max_mcmc_step> => nb_mcmc_moves_per_block={nb_mcmc_moves_per_block}.')

        if (ccohesin_fixed_sides_conf == "opposite") and (nb_ccohesin_same_anchor == 1):
            update_odd = None
        else:
            update_odd = True
            nb_mcmc_moves_per_block = nb_mcmc_moves_per_block*2
                        
    
    for cur_step in range(0, nb_blocks):
        logging.info(f'block index={cur_step}')

        # get the start datetime of each block
        st_block = datetime.datetime.now()

        sim.run(block_size)

        # get the end datetime
        et_block = datetime.datetime.now()

        # get execution time
        elapsed_time_block = et_block - st_block
        logging.info(f'Execution time of block {cur_step}: {elapsed_time_block} seconds')

        logging.info(f'kin temp = {thermodynamic_properties.kinetic_temperature}, E_P/N = {thermodynamic_properties.potential_energy / n_tot}')
        
        
        if (ccohesin_motility == "dynamic") and (cur_step < (nb_blocks-1)):
            
            # get the start datetime of each 1D mcmc
            st_block = datetime.datetime.now()
        
            frame = sim.state.get_snapshot() 
            box = freud.box.Box.from_box(frame.configuration.box)
            particles_position = box.unwrap(
                frame.particles.position,
                frame.particles.image
            )

            polymer_bonds = frame.bonds.group[frame.bonds.typeid==0]
            ccohesin_bonds = frame.bonds.group[frame.bonds.typeid==1]
            
            new_ccohesin_bonds = ccohesin_bonds.copy()
            
            for i in range(nb_mcmc_moves_per_block):
                new_ccohesin_bonds = mc_cohesin_move(
                    cohesin_bonds=new_ccohesin_bonds, 
                    particles_position=particles_position, 
                    chain_length=chain_length,
                    sliding_links_configuration=sliding_links_configuration,
                    update_odd=update_odd, 
                    max_step=max_mcmc_step,
                    cohesin_bond_k=ccohesin_bond_k,
                    cohesin_bond_length=ccohesin_bond_length,
                    )
                
                if update_odd is not None:
                    update_odd = not update_odd

            frame = sim.state.get_snapshot()
            frame.bonds.group[frame.bonds.typeid==1] = new_ccohesin_bonds
            sim.state.set_snapshot(frame)
            
            # get the end datetime
            et_block = datetime.datetime.now()

            # get execution time
            elapsed_time_1d_mcmc = et_block - st_block
            logging.info(f'Execution time of 1D MCMC {cur_step}: {elapsed_time_1d_mcmc} seconds')
        
        
    # get the end datetime
    et_all_blocks = datetime.datetime.now()

    # get execution time
    elapsed_time_all_blocks = et_all_blocks - st_all_blocks
    logging.info(f'Execution time of all blocks: {elapsed_time_all_blocks} seconds')
   
    
def main():

    import argparse
    import ast
    import yaml
    

    parser = argparse.ArgumentParser(
        description="Runs the simulations for misaligned static cohesion" \
        "When parsing arguments, priority is given as following:" \
        "1) argument specified from command-line," \
        "2) argument specified in the YAML config file," \
        "3) default specified in the parser.",
    )
    
    ######## Input, output, cluster and general script args ######################################################
    
    parser.add_argument(
        '--config_file',
        default=None,
        metavar="FILE",
        type=str,       
        help="Path to config file *.yaml." \
        "When parsing arguments, priority is given as following:" \
        "1) argument specified from command-line," \
        "2) argument specified in the YAML config file," \
        "3) default specified in the parser.",
    )
    
    parser.add_argument(
        '--outdir',
        default=None,
        help='Path to the output folder.' 
    )
    
    parser.add_argument(
        '--create_new_outdir',
        type=bool,
        default=False,
        help='Whether creating a new output folder. If <outdir> is also provided, the newly created output folder will be located inside <outdir>.' 
    )
        
    parser.add_argument(
        '--overwrite',
        type=bool,
        default=False,
        help='If True, in case an output folder with the same name already exists, the script will overwrite it with the new results. default=False' 
    )
    
    parser.add_argument(
        '--replicate_id',
        type=int,
        default=os.getenv("SLURM_ARRAY_TASK_ID"),
        help='ID of the replicate. Useful in case simulations of several replicates of the same system are launched. default=None' 
    )
    
    parser.add_argument(
        '--logfilename',
        default=None,
        help='Provide the filename which logging events are recorded to. default=None' 
    )
    
    parser.add_argument(
        '--loglevel',
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        help='Provide logging level. Example --loglevel debug. default=info. choices=["debug", "info", "warning", "error", "critical"]', 
    )
    
    ######## Polymer simulation args ######################################################

    parser.add_argument(
        '--init_config',
        type=int,
        default=None,
        help='File storing the initial configuration of the system.' \
        'Useful when one wants to restart a simulation by providing the configuration of the system at a certain timepoint.' \
        'If no file is provided, the script will initiate the system. default=None' 
    )
    
    parser.add_argument(
        '--chain_length',
        type=int,
        help='Number of monomers per chain.' 
    )
    
    parser.add_argument(
        '--bp_monomer',
        type=int,
        default=200,
        help='Number of base pairs per monomer. default=200' 
    )
    
    parser.add_argument(
        '--nb_blocks',
        type=int,
        help='Number of configurations to be stored. default=100000' 
    )
    
    parser.add_argument(
        '--block_size',
        type=int,
        default=100000,
        help='Number of steps after which a system configuration is stored.' 
    )
    
    parser.add_argument(
        '--dt',
        type=float,
        default=0.05,
        help='Integrator time step size. default=0.05' 
    )
    
    parser.add_argument(
        '--density',
        type=float,
        default=0.1,
        help='Density of the system. default=0.1' 
    )
    
    parser.add_argument(
        '--A_dpd',
        type=float,
        default=10,
        help='Force coefficient for DPD dynamics. default=10' 
    )
    
    parser.add_argument(
        '--gamma_dpd',
        type=float,
        default=10,
        help='Friction coefficient for DPD dynamics. default=10' 
    )
    
    parser.add_argument(
        '--ccohesin_bond_k',
        type=float,
        default=100,
        help='Potential constant of cohesive bonds between two particles on the two different chains. default=100' 
    )
    
    parser.add_argument(
        '--ccohesin_bond_length',
        type=float,
        default=1,
        help='Rest length of cohesive bonds between two particles on the two different chains. default=1' 
    )
    
    parser.add_argument(
        '--polymer_bond_k',
        type=float,
        default=100,
        help='Potential constant of bonds between two consecutive particles of the same chain. default=100' 
    )
    
    parser.add_argument(
        '--polymer_bond_length',
        type=float,
        default=1,
        help='Rest length of cohesive bonds between two consecutive particles of the same chain. default=1' 
    )
        
    ######## Cohesive links args ######################################################
    
    parser.add_argument(
        '--ccohesin_motility',
        choices=['static', 'dynamic'],
        help='Defines whether cohesive links are movable. choices=["static", "dynamic"]',
    )
    
    parser.add_argument(
        '--ccohesin_nb_fixed_sides',
        choices=[0, 1, 2],
        help='Defines how many sides of a cohesive link are movable. choices=[0=both sides can move, 1=one side is fixed and the other one can move, 2=both sides are fixed and the cohesive link is static]',
    )
    
    parser.add_argument(
        '--ccohesin_fixed_sides_conf',
        choices=["random", "opposite", "same"],
        help='Defines the fixed side of cohesive links, in case they are just 1 per cohesive link. Required in case --ccohesin_motility="dynamic" and --ccohesin_nb_fixed_sides=1. choices=["random"=random side is fixed, "opposite"=the fixed side alternates between consecutive links, "same"=all links are fixed on the same side]',
    )
    
    parser.add_argument(
        '--freq_ccohesin_bp',
        type=int,
        help='Average separation (in base pairs!) between two cohesive cohesin links.' 
    )
        
    parser.add_argument(
        '--bp_dist_ccohesin_anchors_01',
        type=int,
        help='When cohesive links are distributed as pairs instead of single links with one side fixed, this argument defines the average base pairs separation between two fixed sides located on chain 0 and 1, respectively, of two consecutive links.' 
    )
    
    parser.add_argument(
        '--bp_dist_ccohesin_anchors_10',
        type=int,
        help='When cohesive links are distributed as pairs instead of single links with one side fixed, this argument defines the average base pairs separation between two fixed sides located on chain 1 and 0, respectively, of two consecutive links.' 
    )
    
    parser.add_argument(
        '--ccohesin_distribution',
        choices=["expon", "linspace"],
        help='Distribution of cohesive links separation. choices=["expon"=separation is exponentially distributed, "linspace"=all links are evenly separated, None=used in case of static cohesive links, where only the misalignment distribution has to be defined (see below)]' 
    )

    parser.add_argument(
        '--ccohesin_misalignment_distribution_bp',
        type=ast.literal_eval,
        help='Nested dict of the cohesive cohesin bonds parameters in base pairs in the format:' \
            '{ <scipy.stats.distribution_name>:' \
                    '{"params"}: {<param_name>: <param_value>},' \
                    '{"population_frac"}: <fraction of links which misalignment follows the distribution>,' \
            ' <scipy.stats.distribution_name>: ...}.'
    )

    parser.add_argument(
        '--ccohesin_anchor_diffusion_distribution_bp',
        type=ast.literal_eval,
        help='Nested dict of the cohesive cohesin bonds parameters in base pairs in the format:' \
            '{ <scipy.stats.distribution_name>:' \
                    '{"params"}: {<param_name>: <param_value>},' \
                    '{"population_frac"}: <fraction of links which misalignment follows the distribution>,' \
            ' <scipy.stats.distribution_name>: ...}.'
    )
        
    parser.add_argument(
        '--nb_mcmc_moves_per_block',
        type=int,
        help='If --ccohesin_motility == "dynamic", this arg defines the number of 1D moves each cohesive cohesin link has to do between two equilibrations of 3D conformations.' 
    )
    
    parser.add_argument(
        '--max_mcmc_step',
        type=int,
        help='If --ccohesin_motility == "dynamic", this arg defines the max 1D step a cohesive cohesin link can do.' 
    )
    
    parser.add_argument(
        '--add_boundaries',
        type=bool,
        default=False,
        help='If --ccohesin_motility == "dynamic", this arg defines if static links at the boundaries should be added.' 
    )
            
    parser.add_argument(
        '--nb_ccohesin_same_anchor',
        type=int,
        default=1,
        help='If --ccohesin_motility == "dynamic" & --ccohesin_nb_fixed_sides==1 & --ccohesin_fixed_sides_conf=="opposite", this arg defines how many links should be anchored on the same side before changing the anchoring side to the opposite chain.' 
    )
    
    parser.add_argument(
        '--chain_first_anchor',
        type=int,
        default=None,
        choices=[None, 0, 1],
        help='If --ccohesin_motility == "dynamic" & --ccohesin_nb_fixed_sides==1, this arg defines on which chain to put the first cohesin fixed side. If --ccohesin_fixed_sides_conf=="opposite", the cohesin fixed sides will be put in an alternate way starting with the chain defined by this argument. If --ccohesin_fixed_sides_conf=="same", all cohesin fixed sides will be on the chain defined by this argument.' 
    )
    
    #############################################################################################
        
    args = parser.parse_args()
    
    
    ######## Check arguments #####################################################################
    
    # If a YAML config file is provided, load arguments values from it and override their parser default values
    if args.config_file is not None:
        
        with open(args.config_file,'r') as f: 
            config = yaml.safe_load(f)    
        parser.set_defaults(**config)
    
        # Reload arguments to override config file values in case some command line  values are provided manually
        args = parser.parse_args()
        
    if (args.outdir is None) or (args.create_new_outdir):
        args.outdir = f'{str(args.outdir or ".")}/{create_outdirname(vars(args))}/{args.replicate_id if args.replicate_id is not None else ""}'
        args.create_new_outdir = False
        
    args.outdir = pathlib.Path(args.outdir)
        

    check_args(
        bp_monomer=args.bp_monomer,
        ccohesin_motility=args.ccohesin_motility,
        ccohesin_nb_fixed_sides=args.ccohesin_nb_fixed_sides,
        ccohesin_fixed_sides_conf=args.ccohesin_fixed_sides_conf,
        freq_ccohesin_bp=args.freq_ccohesin_bp,
        bp_dist_ccohesin_anchors_01=args.bp_dist_ccohesin_anchors_01,
        bp_dist_ccohesin_anchors_10=args.bp_dist_ccohesin_anchors_10,
        ccohesin_distribution=args.ccohesin_distribution,
        ccohesin_misalignment_distribution_bp=args.ccohesin_misalignment_distribution_bp,
        nb_mcmc_moves_per_block=args.nb_mcmc_moves_per_block,
        max_mcmc_step=args.max_mcmc_step,
        chain_first_anchor=args.chain_first_anchor,
    )
    
    ##############################################################################################
    
    if args.outdir.exists():
        if not args.overwrite:
            raise Exception(f'The output folder {args.outdir} already exists! If you want to overwrite it, pass overwrite=True as command line argument.')
    else:
        args.outdir.mkdir(parents=True)
    
    # Write updated YAML config file
    with open(f'{str(args.outdir).rstrip("/").rstrip(str(args.replicate_id if args.replicate_id is not None else ""))}/sim_parameters.yaml','w') as f: 
        yaml.dump(vars(args), f, default_flow_style=False)
        
    # Setup logging
    logging.basicConfig( filename=f'{args.outdir}/{args.logfilename}', filemode='w', encoding='utf-8', level=args.loglevel.upper() )
    logging.info( f'Logging level setup to {args.loglevel.upper()}' )

    logging.info(f"Configuration parameters: \n {vars(args)}")
    
        
    run_sim(**vars(args))
    
    
if __name__ == "__main__":
    main()

    
    
### TODO: 
### 1) restart sim from old trajectory file
