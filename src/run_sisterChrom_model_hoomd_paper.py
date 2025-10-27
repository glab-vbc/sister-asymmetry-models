#!/usr/bin/env python

import itertools
import subprocess
import pprint
import yaml
import pathlib
import re
import os
  
# modified from https://stackoverflow.com/questions/70178819/combination-of-nested-dictionaries-with-arbitrary-lengths-in-python
def c_prod(d):
    if isinstance(d, list):
        for i in d:
            yield from ([i] if not isinstance(i, (dict, list)) else c_prod(i))
    elif isinstance(d, dict):
        for i in itertools.product(*map(c_prod, d.values())):
            yield dict(zip(d.keys(), i))
    else:
        yield d
        
        
        

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

            #if parameters["ccohesin_fixed_sides_conf"] is not None:
            #    outdirname = outdirname + f'_ccohesinFixedSidesConf-{parameters["ccohesin_fixed_sides_conf"]}'

            if parameters["ccohesin_distribution"] is not None:
                outdirname = outdirname + f'_ccohesinDistribution-{parameters["ccohesin_distribution"]}'
                
            if "ccohesin_misalignment_distribution_bp" in parameters.keys() and (parameters["ccohesin_misalignment_distribution_bp"] is not None): 
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
            if "ccohesin_anchor_diffusion_distribution_bp" in parameters.keys():
                for distribution_name, distribution_params in parameters['ccohesin_anchor_diffusion_distribution_bp'].items():

                    distribution_name = distribution_name.replace('_', '')
                    outdirname = outdirname+f"_ccohesinAnchorDiffusionDistributionBp-{distribution_name}"

                    for param,value in distribution_params["params"].items():

                        param = re.sub(r'(_[a-z])', lambda pat: pat.group(1).upper().strip("_"), param)
                        outdirname = outdirname+"_"+"-".join([param, str(value)])

                    outdirname = outdirname+"_frac-"+str(distribution_params["population_frac"])
                    
            if ("diffusive_links_between_semianchored_pairs" in parameters.keys()) and (parameters["diffusive_links_between_semianchored_pairs"] is not None) and (parameters["diffusive_links_between_semianchored_pairs"]):
                outdirname = outdirname + f'_nbDiffusiveCcohesin01-{parameters["nb_diffusive_ccohesin_01"]}'
                outdirname = outdirname + f'_nbDiffusiveCcohesin10-{parameters["nb_diffusive_ccohesin_10"]}'
   
            if "nb_ccohesin_same_anchor" in parameters.keys():
                outdirname = outdirname + f'_nbCcohesinSameAnchor-{parameters["nb_ccohesin_same_anchor"]}'
            if "add_boundaries" in parameters.keys():
                outdirname = outdirname + f'_addBoundaries-{parameters["add_boundaries"]}'
            if ("unloading_fraction" in parameters.keys()) and (parameters["unloading_fraction"] is not None) and (parameters["unloading_fraction"]):
                outdirname = outdirname + f'_unloadingFrac-{parameters["unloading_fraction"]}'
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

'''


def create_outdirname(parameters):
    
    if parameters["init_config"] is None:
    
        outdirname = f'model-{parameters["ccohesin_motility"]}'

        if parameters["ccohesin_motility"] == "static":

            outdirname = outdirname + f'_freqCcohesinBp-{parameters["freq_ccohesin_bp"]}'

            #for distribution_name, distribution_params in parameters['ccohesin_misalignment_distribution_bp'].items():
            for distribution_id, distribution in parameters['ccohesin_misalignment_distribution_bp'].items():

                #distribution_name = distribution_name.replace('_', '')
                distribution_name = distribution['name']
                outdirname = outdirname+f"_ccohesinMisalignmentDistributionBp-{distribution_name}"

                #for param,value in distribution_params["params"].items():
                for param,value in distribution["params"].items():

                    param = re.sub(r'(_[a-z])', lambda pat: pat.group(1).upper().strip("_"), param)
                    outdirname = outdirname+"_"+"-".join([param, str(value)])

                #outdirname = outdirname+"_frac-"+str(distribution_params["population_frac"])
                outdirname = outdirname+"_frac-"+str(distribution["population_frac"])

        elif parameters["ccohesin_motility"] == "dynamic":

            outdirname = outdirname + f'_ccohesinNbFixedSides-{parameters["ccohesin_nb_fixed_sides"]}'

            if parameters["ccohesin_fixed_sides_conf"] is not None:
                outdirname = outdirname + f'_ccohesinFixedSidesConf-{parameters["ccohesin_fixed_sides_conf"]}'

            if parameters["ccohesin_distribution"] is not None:
                outdirname = outdirname + f'_ccohesinDistribution-{parameters["ccohesin_distribution"]}'
                
            if parameters["ccohesin_misalignment_distribution_bp"] is not None: 
                #for distribution_name, distribution_params in parameters['ccohesin_misalignment_distribution_bp'].items():
                for distribution_id, distribution in parameters['ccohesin_misalignment_distribution_bp'].items():

                    #distribution_name = distribution_name.replace('_', '')
                    distribution_name = distribution['name']
                    outdirname = outdirname+f"_ccohesinMisalignmentDistributionBp-{distribution_name}"

                    #for param,value in distribution_params["params"].items():
                    for param,value in distribution["params"].items():

                        param = re.sub(r'(_[a-z])', lambda pat: pat.group(1).upper().strip("_"), param)
                        outdirname = outdirname+"_"+"-".join([param, str(value)])

                    #outdirname = outdirname+"_frac-"+str(distribution_params["population_frac"])
                    outdirname = outdirname+"_frac-"+str(distribution["population_frac"])

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
'''
        
def launch_job_arrays_all_param_comb(script_file, 
                                     conda_env, 
                                     config_file_all_param_comb,
                                     outdir,
                                     first_replicate_id,
                                     time,
                                     qos,
                                     nb_nodes=1,
                                     partition='g',
                                     gres='gpu:1',
                                     cpus_per_task=1,
                                     mem_per_cpu='8G',
                                     constraint="\[g4\|g2\|g1\]",
                                     overwrite_outdir=False,
                                     dry_run=False):
    
    with open(config_file_all_param_comb,'r') as f: 
        cfg = yaml.safe_load(f)
    print("\nConfiguration parameters to combine:")
    pprint.pprint(cfg)
    
    nb_replicates = cfg["nb_replicates"]
    del cfg["nb_replicates"]

    list_param_comb = [param_comb for param_comb in c_prod(cfg)]

    N_PARAM_COMBOS = len(list_param_comb)
    print(f'\nSweeping over {N_PARAM_COMBOS} parameter combinations!')
    
    for i, param_comb in enumerate(list_param_comb):
        print("----------------")
        print("\nPARAM COMB ",i)
        pprint.pprint(param_comb)

        launch_job_array(script_file,
                         conda_env, 
                         param_comb,
                         outdir,
                         nb_replicates,
                         first_replicate_id,
                         time,
                         qos,
                         nb_nodes,
                         partition,
                         gres,
                         cpus_per_task,
                         mem_per_cpu,
                         constraint,
                         overwrite_outdir,
                         dry_run)
    
def launch_job_array(script_file, 
                     conda_env, 
                     parameters_combination,
                     outdir,
                     nb_replicates,
                     first_replicate_id,
                     time,
                     qos,
                     nb_nodes=1,
                     partition='g',
                     gres='gpu:1',
                     cpus_per_task=1,
                     mem_per_cpu='8G',
                     constraint="\[g4\|g2\|g1\]",
                     overwrite_outdir=False,
                     dry_run=False):

    if outdir is not None:
        outdir = pathlib.Path(outdir)
    else:
        outdir = pathlib.Path(os.getcwd())

    out_dir_name = create_outdirname(parameters_combination)
    #out_dir_name = out_dir_name+"_A-"+str(parameters_combination["A_dpd"])+"_gamma-"+str(parameters_combination["gamma_dpd"])+"_density-"+str(parameters_combination["density"])+"_bondK-"+str(parameters_combination["polymer_bond_k"])+"_integrator-dpd_startconfig-100" #_force-lj84RepOnly_epsilon-1_sigma-1"
    #out_dir_name = out_dir_name+"_A-"+str(parameters_combination["A_dpd"])+"_gamma-"+str(parameters_combination["gamma_dpd"])+"_ccohesBondK-"+str(parameters_combination["ccohesin_bond_k"])
    #out_dir_name = out_dir_name+"_A-"+str(parameters_combination["A_dpd"])+"_gamma-"+str(parameters_combination["gamma_dpd"])
    #out_dir_name = out_dir_name+f'_logbuffer-{parameters_combination["logbuffer"]}_scratchcbe'

    out_dir = outdir / out_dir_name

    if out_dir.exists():
        if not overwrite_outdir:
            raise Exception(f'The output folder {out_dir} already exists! If you want to overwrite it, pass overwrite_outdir=True as command line argument.')
    else:
        out_dir.mkdir(parents=True)
        
    #print(f'Output folder: {out_dir}')
    
    config_file = f"{out_dir}/config.yaml"
    with open(config_file, 'w',) as f :
        yaml.dump(parameters_combination, f, default_flow_style=False)
    
    python_cmd = [
        'conda',
        'run',
        '-n',
        conda_env,
        'python',
        script_file,
        '--config_file',
        config_file,
    ]

    '''
    for name, value in param_comb.items():
        if value is not None:
            python_cmd.append(name)
            if bool(str(value)):

                if isinstance(value, dict):
                    #value = str(value).replace("'", '"')
                    #value = value.replace('"', '\"')
                    #value = {'\"'+str(k)+'\"':v for k,v in value.items()}
                    #value = value.replace('"', "'")
                    #value = value.replace("{", "\{")
                    #value = value.replace("}", "\}")
                    #value = value.replace(":", "\:")
                    #value = "'" + str(value).replace("'", '"').replace('"', '\"') + "'"
                    python_cmd.append(str(value))
                else:
                    python_cmd.append(str(value))

    python_cmd.append("--replicate_id")
    #python_cmd.append("\${SLURM_ARRAY_TASK_ID}")
    python_cmd.append('0')
    '''
    python_cmd = ' '.join(python_cmd)
    
    slurm_array_task_id = "\${SLURM_ARRAY_TASK_ID}"

    #job_name = []
    #job_name.append('dpd-start100')
    #job_name.append(f'A-{parameters_combination["A_dpd"]}')
    #job_name.append(f'gamma-{parameters_combination["gamma_dpd"]}')
    #job_name.append(f'density-{parameters_combination["density"]}')
    #job_name.append(f'k-{parameters_combination["ccohesin_bond_k"]}')
    #for distr,params in parameters_combination["ccohesin_misalignment_distribution_bp"].items():
    #    job_name.append(f"{distr}")
    #    for param,value in params["params"].items():
    #        job_name.append(f"{param}-{value}")
        #job_name.append(f'frac-{params["population_frac"]}')
    #job_name.append(f'freq-{parameters_combination["freq_ccohesin_bp"]}')
    #job_name = '_'.join(job_name)
    job_name = out_dir_name
        
    sbatch_cmd = ['sbatch',
                  f'--job-name={job_name}',
                  f'--nodes={nb_nodes}',
                  f'--partition={partition}',
                  f'--gres={gres}',
                  f'--mem-per-cpu={mem_per_cpu}',
                  f'--qos={qos}',
                  f'--time={time}',
                  f'--constraint={constraint}',
                  f'--cpus-per-task={cpus_per_task}',
                  f'--array={first_replicate_id}-{nb_replicates-1+first_replicate_id}',
                  f'--output={out_dir}/%a/{parameters_combination["logfilename"]}',
                  f'--wrap="{python_cmd} --outdir {out_dir}/{slurm_array_task_id}/"',
                 ]
    sbatch_cmd = ' '.join(sbatch_cmd)

    #cmd = sbatch_cmd + ['"',] + python_cmd + ['"',]

    #cmd = sbatch_cmd + python_cmd

    print(f'Executing command: {sbatch_cmd}' )

    if not dry_run:
        subprocess.call(sbatch_cmd, shell=True)
    
    
    
def main():
    
    # python /users/flavia.corsi/src/run_sisterChrom_model_hoomd_paper.py --config_file_all_param_comb src/config.yaml --script_file /users/flavia.corsi/src/sisterChrom_model_hoomd_paper.py --outdir /groups/goloborodko/projects/lab/sisterChromatids2020/static_cohesion_hoomd_paper/ --time 2-00:00:00 --qos g_medium --overwrite_outdir --constraint "\[g4\|g2\]"

    import argparse
    import ast

    parser = argparse.ArgumentParser(description="Launch the parameter sweep for the simulations of misaligned static cohesion")
    
    parser.add_argument(
        '--script_file',
        help='Python script with the simulation code.' 
    )
    
    parser.add_argument(
        '--conda_env',
        default='hoomd-py39',
        help='Name of the Conda environment to use for the simulations.' 
    )
    
    parser.add_argument(
        '--config_file_all_param_comb',
        help='Configuration file *.yml to compute all the combinations of parameters of interest and pass them to the script to run the simulation.' 
    )
    
    parser.add_argument(
        '--outdir',
        help='Root folder where the output subfolders (one subfolder per combination of parameters) are placed.' 
    )
    
    parser.add_argument(
        '--first_replicate_id',
        type=int,
        default=0,
        help='ID of the first replicate. Default is 0' 
    )
    
    parser.add_argument(
        '--overwrite_outdir',
        action="store_true",
        help='If True, the script will overwrite the output folder in case it already exists. default=False' 
    )
     
    parser.add_argument(
        '--time',
        type=str,
        help='Max time to be allocated for each job. Format: days-hours:minutes:seconds. Example: 02-12:30:00' 
    )
    
    parser.add_argument(
        '--qos',
        choices=['g_short', 'g_medium', 'g_long'],
        help='Resource limits. choices=["g_short", "g_medium", "g_long"].' \
        '"g_short" = 8 hours (08:00:00) limit; "g_medium" = 2 days (2-00:00:00) limit; "g_long" = 14 days (14-00:00:00) limit' 
    )
    
    parser.add_argument(
        '--nb_nodes',
        default=1,
        help='Number of nodes to be allocated for each job.' 
    )
    
    parser.add_argument(
        '--partition',
        default="g",
        choices=["c","g","m"],
        help='Partition name. choices=["c","g","m"]. "c"=CPU, "g"=GPU, "m"=MEMORY. default="g"' 
    )
    
    parser.add_argument(
        '--gres',
        default='gpu:1',
        help='GPU resources, in case a GPU node is selected. default="gpu:1"' 
    )
    
    parser.add_argument(
        '--cpus_per_task',
        default=1,
        help='Number of CPUs per task. default=1' 
    )
    
    parser.add_argument(
        '--mem_per_cpu',
        default="16G",
        help='Memory limit per CPU. default=8G' 
    )
    
    parser.add_argument(
        '--constraint',
        default="\[g4\|g2\|g1\]",
        help='Constraint which node(s) to use. One (ex. g1) or multiple nodes (\[g4\|g2\...\]) can be specified.' \
        'default="\[g4\|g2\|g1\]" i.e allow only nodes g4 or g2 or g1. The escape is needed to the shell to interprest the string correctly.' 
    )
        
    parser.add_argument(
        '--dry_run',
        action="store_true",
        help='Perform a dry run of the script. default=False' 
    )
     
    args = parser.parse_args()
    
    
    launch_job_arrays_all_param_comb(script_file=args.script_file, 
                                     conda_env=args.conda_env, 
                                     config_file_all_param_comb=args.config_file_all_param_comb,
                                     outdir=args.outdir,
                                     first_replicate_id=args.first_replicate_id,
                                     time=args.time,
                                     qos=args.qos,
                                     nb_nodes=args.nb_nodes,
                                     partition=args.partition,
                                     gres=args.gres,
                                     cpus_per_task=args.cpus_per_task,
                                     mem_per_cpu=args.mem_per_cpu,
                                     constraint=args.constraint,
                                     overwrite_outdir=args.overwrite_outdir,
                                     dry_run=args.dry_run)
    
    
if __name__ == "__main__":
    main()