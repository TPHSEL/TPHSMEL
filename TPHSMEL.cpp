// TPHDSL.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <fstream>
#include <sstream>
#include <ctime>
#include <cmath>
#include <cstring>
using namespace std;

#define VHV 10000000000000	//very high value
#define NUMOPRTS 6	//number of mathematical operators
#define NITRS 13			//number of iterations
#define MAXFEATURES 150	//maximum allowable number of features
#define PRECISION 0.00000000000001
#define MAXHIERARCHIES 200
#define MAXCLASSES 10
#define MAXSAMPLES 8000
#define MAXFOLDS 10

unsigned short minimum_generations;
int generation, terminal_generation;
unsigned short model_sample_fitness[MAXHIERARCHIES], subset[MAXSAMPLES], class_fitness[31], ultimate_fitness;
double set_part[31], model_set_part[MAXHIERARCHIES], sample_fitness[31], model_prb_cover[MAXHIERARCHIES];
double model_val_max[MAXHIERARCHIES][MAXCLASSES], model_val_min[MAXHIERARCHIES][MAXCLASSES], model_val_mean[MAXHIERARCHIES][MAXCLASSES];
double feature[MAXFEATURES][MAXSAMPLES], feature_set[MAXFEATURES][MAXSAMPLES], selected_feature[MAXFEATURES][MAXSAMPLES];
char this_class[MAXCLASSES][100], class_name[MAXCLASSES][100];
unsigned short class_is[10][MAXSAMPLES], class_found[MAXSAMPLES], class_size[MAXCLASSES], chrom_length;
unsigned short tr_size, ts_size, num_samples, num_classes, elite, level, num_features;
unsigned short trs[4000], tss[4000];
double average[MAXCLASSES], std_dvt[MAXCLASSES], maximum[MAXCLASSES], minimum[MAXCLASSES], training_size;
double chromosome_selected[31][410], chromosome[31][410], phromosome[MAXHIERARCHIES][410], prb_cover[31];
unsigned short run_correct, run_total, population, tree[2 * MAXFEATURES][3], num_folds;
unsigned short test_level[MAXSAMPLES], vote[MAXCLASSES][MAXSAMPLES], weighted_vote[MAXCLASSES][MAXSAMPLES];

double __min(double var1, double var2)
{
        if(var1 < var2)return var1;
        return var2;
}

double __max(double var1, double var2)
{
        if(var1 > var2)return var1;
        return var2;
}

class Conflict_Resolution {
private:
	// some data
	unsigned short similarity_index;
	unsigned short numsamples[MAXCLASSES];
	bool class_match;
	// some functions
public:
	//to check if meta class is needed
	bool similarity_index_check(unsigned short* sample1, unsigned short* sample2)
	{
		unsigned short spl1, spl2, spl;
		for (spl1 = 1; spl1 < tr_size; spl1++)
		{
			for (spl2 = spl1 + 1; spl2 <= tr_size; spl2++)
			{
				class_match = false;
				for (spl = 1; spl <= class_is[0][trs[spl1]]; spl++)
				{
					if (class_is[1][trs[spl2]] == class_is[spl][trs[spl1]])
					{
						class_match = true;
						break;
					}
				}
				if(class_match) continue;
				similarity_index = 0;
				for (unsigned short ftr = 1; ftr <= num_features; ftr++)
				{
					if (abs(feature[ftr][trs[spl1]] - feature[ftr][trs[spl2]]) < PRECISION)
						similarity_index++;
				}
				if (similarity_index == num_features)
				{
					*sample1 = spl1;
					*sample2 = spl2;
					return true;
				}
			}
		}
		return false;
	}

	//resolving the conflict
	void conflict_resolution(unsigned short spl1, unsigned short spl2)
	{
		class_is[0][trs[spl1]]++;
		class_is[class_is[0][trs[spl1]]][trs[spl1]] = class_is[1][trs[spl2]];
		class_is[0][trs[spl2]]++;
		class_is[class_is[0][trs[spl2]]][trs[spl2]] = class_is[1][trs[spl1]];
	}

	//computing samples in each class
	unsigned short* computing_samples()
	{
		//initialisation
		for (unsigned short cls = 1; cls <= num_classes; cls++)
			numsamples[cls] = 0;
		//number of samples in each class in training set
		for (unsigned short spl = 1; spl <= tr_size; spl++)
			numsamples[class_is[1][trs[spl]]]++;
		return numsamples;
	}
};

class Evaluation {
private:
	// some data
	double wtd_ftr[2 * MAXFEATURES];
	// some functions
public:
	//operations on features
	double evaluate(unsigned short a, double b, double c)
	{
		if (b == 0) return c;		//trim the branch in case feature doesn't exist 
		if (c == 0) return b;      //same as above
		switch (a)
		{
		case 1: return b + c;
		case 2: return b - c;
		case 3: return b * c;
		case 4: return b / c;
		case 5: return __max(b, c);
		case 6: return __min(b, c);
		}
		return 0;
	}

	//genotype to phenotype translation
	double evaluate(double* _romosome, unsigned short spl)
	{
		unsigned short ftr_id, gene;
		for (gene = 2 * num_features - 1; gene >= num_features; gene--)
		{
			ftr_id = _romosome[gene];
			wtd_ftr[gene] = _romosome[tree[gene][1]] * feature[ftr_id][spl];
		}
		for (; gene >= 1; gene--)
			wtd_ftr[gene] = evaluate(_romosome[gene], wtd_ftr[tree[gene][1]], wtd_ftr[tree[gene][2]]);

		return wtd_ftr[1];
	}
};

class Statistics {
private:
	// some data
	unsigned short spl;
	double sigma;
	// some functions
public:
	// finding maximum value of the class
	double max(double* value, unsigned short id, unsigned short& smpl)
	{
		double mxm(-VHV);
		for (spl = 1; spl <= tr_size; spl++)
		{
			if (class_is[1][trs[spl]] != id) continue;
			if (value[trs[spl]] > mxm)mxm = value[trs[spl]];
		}
		return mxm;
	}
	// finding minimum value of the class
	double min(double* value, unsigned short id, unsigned short& smpl)
	{
		double mnm(VHV);
		for (spl = 1; spl <= tr_size; spl++)
		{
			if (class_is[1][trs[spl]] != id) continue;
			if (value[trs[spl]] < mnm)mnm = value[trs[spl]];
		}
		return mnm;
	}
	//finding mean of the sample
	double mean(double* value, unsigned short id, unsigned short& smpl)
	{
		double sum(0.0);
		for (spl = 1; spl <= tr_size; spl++)
		{
			if (class_is[1][trs[spl]] != id) continue;
			sum += value[trs[spl]];
			smpl++;
		}
		if (smpl == 0) return 0;        //there is no member of the class
		return sum / double(smpl);
	}
	//standard deviation
	double standard_deviation(double* value, double mean, unsigned short id, unsigned short smpl)
	{
		if (smpl == 0) return 0;	//there is no member of the class
		double sum(0.0);
		for (spl = 1; spl <= tr_size; spl++)
		{
			if (class_is[1][trs[spl]] != id) continue;
			sum += pow(value[trs[spl]] - mean, 2);
		}
		return sqrt(sum / double(smpl));
	}

	//statistical parameters
	void statistical_parameters(double* value)
	{
		unsigned short nspl, cls;
		double mxm[MAXCLASSES]{}, mnm[MAXCLASSES]{};
		//finding minima, maxima and mean of the training set
		for (cls = 1; cls <= num_classes; cls++)
		{
			nspl = 0;
			average[cls] = mean(value, cls, nspl);
			std_dvt[cls] = standard_deviation(value, average[cls], cls, nspl);
			mxm[cls] = max(value, cls, nspl);
			mnm[cls] = min(value, cls, nspl);
			maximum[cls] = __max(average[cls] + std_dvt[cls], mxm[cls]);
			minimum[cls] = __min(average[cls] - std_dvt[cls], mnm[cls]);
		}
	}
};

class Classification {
private:
	// some data
	Statistics stats;
	Evaluation eva;
	Conflict_Resolution cr;
	// some functions
public:
	//classification
	unsigned short find_class(double* prb, double set_partition)
	{
		unsigned short cls, id, found_class;
		for (cls = 1; cls <= num_classes; cls++)
		{
			for (id = 1; id <= num_classes; id++)
			{
				if (id == cls) continue;
				if (prb[cls] > prb[id] + set_partition) continue;
				found_class = num_classes + 1;
				break;
			}
			if (id > num_classes)
			{
				found_class = cls;
				break;
			}
		}
		return found_class;
	}

	bool two_values_are_equal(double q1, double q2)
	{
		if (q1 <= q2+PRECISION and q1 >= q2-PRECISION)
			return true;
		return false;
	}

	//finding relativity
	double probability(double dim, double max, double min, double mean)
	{
		if (two_values_are_equal(dim, max) and two_values_are_equal(dim, min)) return 1; //all the class members and sample under testing are equal
		if (two_values_are_equal(min, max)) return -abs(mean-dim); //all the class members are equal but the sample under testing is different
		if (min > max) return -abs(dim);				//no member of the class exists
		if (dim > mean) return (max - dim) / (max - mean);	//if it has above average value
		return (dim - min) / (mean - min);			//if it has bellow average value
	}

	//test set experiments for heirarchical model
	void test_set()
	{
		double value[MAXSAMPLES], prb[MAXHIERARCHIES][MAXCLASSES];
		unsigned short spl, id, lvl, cls;
		for (spl = 1; spl <= ts_size; spl++)
		{
			//cout << "spl: " << spl << endl;
			for (lvl = 1; lvl <= level; lvl++)
			{
				value[tss[spl]] = eva.evaluate(phromosome[lvl], tss[spl]);
				for (id = 1; id <= num_classes; id++)
					prb[lvl][id] = probability(value[tss[spl]], model_val_max[lvl][id], model_val_min[lvl][id], model_val_mean[lvl][id]);
				class_found[tss[spl]] = find_class(prb[lvl], model_set_part[lvl]);
				if (class_found[tss[spl]] <= num_classes)
				{
					vote[class_found[tss[spl]]][tss[spl]] += 1;
					weighted_vote[class_found[tss[spl]]][tss[spl]] += model_sample_fitness[lvl];
					break;
				}
			}
			if (lvl > level)	//if still unclassified
			{
				class_found[tss[spl]] = class_is[1][trs[1]];
				vote[class_found[tss[spl]]][tss[spl]] += 1;
				weighted_vote[class_found[tss[spl]]][tss[spl]] += model_sample_fitness[level];
			}
			test_level[tss[spl]] = lvl;
		}
	}

	//partition of set
	void set_partition()
	{
		unsigned short uc_size(0), spl, guess, ftr, cls, * numsamples; //size of unclassified set
		double prb[MAXCLASSES], value[MAXSAMPLES];
		numsamples = cr.computing_samples();
		for (spl = 1; spl <= tr_size; spl++)
			value[trs[spl]] = eva.evaluate(chromosome[elite], trs[spl]);
		stats.statistical_parameters(value);	//finding minimum, maximum, mean and standard deviation of all samples of the training set
                
		for (spl = 1; spl <= tr_size; spl++)
		{
			for (cls = 1; cls <= num_classes; cls++)
				prb[cls] = probability(value[trs[spl]], maximum[cls], minimum[cls], average[cls]);
			guess = find_class(prb, set_part[elite]);
			if (guess > num_classes)
				trs[++uc_size] = trs[spl];
		}
		level++;			//incrementing hierarchy level
		//saving hierarchy model to phromosome
		for (unsigned short phene = 1; phene <= chrom_length; phene++)
			phromosome[level][phene] = chromosome[elite][phene];
		model_set_part[level] = set_part[elite];
		model_prb_cover[level] = prb_cover[elite];
		model_sample_fitness[level] = tr_size - uc_size;
		for (unsigned short id = 1; id <= num_classes; id++)
		{
			model_val_max[level][id] = maximum[id];
			model_val_min[level][id] = minimum[id];
			model_val_mean[level][id] = average[id];
		}
		if (tr_size == uc_size) level--;
		else tr_size = uc_size;  //reducing training set to unclassified set
	}
};

class Reproduction {
private:
	// some data
	unsigned short gene, gene0;
	// some functions
public:
	//copying back to original solution
	void cloning()
	{
		for (unsigned short chrom = 1; chrom <= population; chrom++)	//copying whole population
			for (unsigned short gene = 1; gene <= chrom_length; gene++)
				chromosome[chrom][gene] = chromosome_selected[chrom][gene];
	}

	//population selection for next generation
	void selection_scheme()
	{
		unsigned short chrom(0), chrom_1, chrom_2, select;
		do
		{
			chrom++;
			if (chrom == elite) select = chrom;	//elite selection	
			else
			{
				chrom_1 = 1 + rand() % population;	//random selection of first chromosome
				chrom_2 = 1 + rand() % population;	//random selection of second chromosome
				if (class_fitness[chrom_1] > class_fitness[chrom_2]) select = chrom_1;
				else if (class_fitness[chrom_1] == class_fitness[chrom_2] and sample_fitness[chrom_1] > sample_fitness[chrom_2]) select = chrom_1;	//hierarchical tournament between randomly selected chromosomes
				else if (class_fitness[chrom_1] == class_fitness[chrom_2] and sample_fitness[chrom_1] > sample_fitness[chrom_2] - PRECISION and prb_cover[chrom_1] > prb_cover[chrom_2]) select = chrom_1;
				else if (class_fitness[chrom_1] == class_fitness[chrom_2] and sample_fitness[chrom_1] > sample_fitness[chrom_2] - PRECISION and prb_cover[chrom_1] > prb_cover[chrom_2] - PRECISION and set_part[chrom_1] < set_part[chrom_2]) select = chrom_1;
				else select = chrom_2;
			}
			for (unsigned short gene = 1; gene <= chrom_length; gene++)
				chromosome_selected[chrom][gene] = chromosome[select][gene];	//copying to temporary variable
		} while (chrom < population);
		cloning();
	}

	//mutation
	void mutation(unsigned short chrom)
	{
		gene = 1 + rand() % chrom_length;	//randomly selecting gene
		if (gene < num_features)
		{
			do
				gene0 = 1 + rand() % NUMOPRTS;
			while (gene0 == chromosome_selected[chrom][gene]);
			chromosome_selected[chrom][gene] = gene0;
		}
		else if (gene < 2 * num_features)
		{
			do
				gene0 = num_features + rand() % num_features;
			while (gene0 == gene);
			swap(chromosome_selected[chrom][gene], chromosome_selected[chrom][gene0]);
		}
		else
		{
			gene0 = rand() % 2;
			if (gene0 == 0)
				chromosome_selected[chrom][gene] -= pow((rand() / double(RAND_MAX)), 1 - (terminal_generation - generation) / double(terminal_generation));
			else chromosome_selected[chrom][gene] += pow((rand() / double(RAND_MAX)), 1 - (terminal_generation - generation) / double(terminal_generation));
		}
	}

	//tree crossover
	void crossover(unsigned short candidate_1, unsigned short candidate_2)
	{
		gene = 2 + rand() % (chrom_length - 1);	//generating random cut point
		if (gene < num_features)	//this is operator
		{
			chromosome_selected[candidate_1][gene] = chromosome[candidate_2][gene];
			chromosome_selected[candidate_2][gene] = chromosome[candidate_1][gene];
		}
		else if (gene < 2 * num_features)	//this is feature
		{
			unsigned short n(0);
			gene0 = num_features;
			for (;; gene0++)
			{
				if (chromosome_selected[candidate_1][gene0] == chromosome[candidate_2][gene])
				{
					swap(chromosome_selected[candidate_1][gene0], chromosome_selected[candidate_1][gene]);
					n++;
				}
				if (chromosome_selected[candidate_2][gene0] == chromosome[candidate_1][gene])
				{
					swap(chromosome_selected[candidate_2][gene0], chromosome_selected[candidate_2][gene]);
					n++;
				}
				if (n == 2) break;
			}
		}
		else	//this is weight
		{
			double arithmatic_cross, greater, lesser;
			greater = __max(chromosome[candidate_1][gene], chromosome[candidate_2][gene]);
			lesser = __min(chromosome[candidate_1][gene], chromosome[candidate_2][gene]);
			arithmatic_cross = rand() / float(RAND_MAX) * (greater - lesser);
			if (chromosome[candidate_1][gene] > chromosome[candidate_2][gene])
			{
				chromosome_selected[candidate_1][gene] = chromosome[candidate_1][gene] - arithmatic_cross;
				chromosome_selected[candidate_2][gene] = chromosome[candidate_2][gene] + arithmatic_cross;
			}
			else
			{
				chromosome_selected[candidate_2][gene] = chromosome[candidate_2][gene] - arithmatic_cross;
				chromosome_selected[candidate_1][gene] = chromosome[candidate_1][gene] + arithmatic_cross;
			}
		}
	}

	//production of next generation
	void next_generation(unsigned short* oprtr_select)
	{
		unsigned short chrom(0), cand(0), candidate[3];
		do
		{
			chrom++;
			if (oprtr_select[chrom] == 0)  //candidate for crossover
			{
				cand++;
				candidate[cand] = chrom;
				if (cand == 2)	//crossover is only possible with two chromosomes
				{
					crossover(candidate[1], candidate[2]);
					cand = 0;
				}
			}
			else if (oprtr_select[chrom] == 1) mutation(chrom);
		} while (chrom < population);
		cloning();
	}

	//selection of genetic operators for the production of next generation
	void operator_selection()
	{
		unsigned short chrom(0), oprtr_select[31];
		do
		{
			chrom++;
			if (chrom == elite) oprtr_select[chrom] = 2;	//elite cloning
			else oprtr_select[chrom] = rand() % 2;	//random selection of operator
		} while (chrom < population);
		next_generation(oprtr_select);
	}
};

class Evolution
{
private:
	// some data
	Evaluation eva;
	// some functions
public:
	//generating random population
	void population_generation()
	{
		bool ins[MAXFEATURES];
		population = __min(__min(chrom_length, tr_size) * 2, 30); //for genetic algorithm
		for (unsigned short chrom = 1; chrom <= population; chrom++)
		{
			for (unsigned short ftr = 1; ftr <= num_features; ftr++)
				ins[ftr] = false;
			for (unsigned short gene = 1; gene <= chrom_length; gene++)
			{
				if (gene < num_features) chromosome[chrom][gene] = 1 + rand() % NUMOPRTS;
				else if (gene < 2 * num_features)
				{
					do
						chromosome[chrom][gene] = 1 + rand() % num_features;
					while (ins[int(chromosome[chrom][gene])]);
					ins[int(chromosome[chrom][gene])] = true;
				}
				else chromosome[chrom][gene] = rand() / float(RAND_MAX);
			}
		}
	}

	//evaluation of fitness of each and every chromosome
	unsigned short fitness_evaluation()
	{
		double value[MAXSAMPLES]{};
		elite = 0;
		double prb[MAXCLASSES]{};
		unsigned short phrom, spl, cls, icls;
		unsigned short guess, numsamps_fit[MAXCLASSES]{}, * numsamples;
		Conflict_Resolution cr;
		Classification clf;
		Statistics stats;
		numsamples = cr.computing_samples();
		for (phrom = 1; phrom <= population; phrom++)
		{
			prb_cover[phrom] = 1.0;
			for (unsigned short i = 1; i <= num_classes; i++) numsamps_fit[i] = 0;
			set_part[phrom] = sample_fitness[phrom] = class_fitness[phrom] = 0;
			for (spl = 1; spl <= tr_size; spl++) value[trs[spl]] = eva.evaluate(chromosome[phrom], trs[spl]);
			stats.statistical_parameters(value);	//finding min, max, mean and std dv of all samples of the training set
			for (spl = 1; spl <= tr_size; spl++)
			{
				for (cls = 1; cls <= num_classes; cls++)
					prb[cls] = clf.probability(value[trs[spl]], maximum[cls], minimum[cls], average[cls]);
				guess = clf.find_class(prb, set_part[phrom]);
				for (icls = 1; icls <= class_is[0][trs[spl]]; icls++)
					if (guess == class_is[icls][trs[spl]])
						break;
				if(icls > class_is[0][trs[spl]] and guess <= num_classes)					//guess isn't successful
					set_part[phrom] = prb[guess] - prb[class_is[1][trs[spl]]] + PRECISION;
			}
			for (spl = 1; spl <= tr_size; spl++)
			{
				for (cls = 1; cls <= num_classes; cls++)
					prb[cls] = clf.probability(value[trs[spl]], maximum[cls], minimum[cls], average[cls]);
				guess = clf.find_class(prb, set_part[phrom]);
				if (guess <= num_classes)
				{
					numsamps_fit[guess]++;
					for (cls = 1; cls <= num_classes; cls++)
					{
						for (icls = 1; icls <= class_is[0][trs[spl]]; icls++)
							if (cls == class_is[icls][trs[spl]])
								continue;
							prb_cover[phrom] = __min(prb[guess] - prb[cls] - set_part[phrom], prb_cover[phrom]);	//if this class is not equal to any of the classes of the sample
					}
				}
			}
			for (cls = 1; cls <= num_classes; cls++)
			{
				if (numsamples[cls] == numsamps_fit[cls]) class_fitness[phrom]++;
				sample_fitness[phrom] += 1 / double(numsamples[cls] - numsamps_fit[cls] + 1);
			}
		}
		for (phrom = 1; phrom <= population; phrom++)
		{
			if (class_fitness[phrom] > class_fitness[elite]) elite = phrom;
			else if (class_fitness[phrom] == class_fitness[elite] and sample_fitness[phrom] > sample_fitness[elite]) elite = phrom;
			else if (class_fitness[phrom] == class_fitness[elite] and sample_fitness[phrom] > sample_fitness[elite] - PRECISION and prb_cover[phrom] > prb_cover[elite]) elite = phrom;
			else if (class_fitness[phrom] == class_fitness[elite] and sample_fitness[phrom] > sample_fitness[elite] - PRECISION and prb_cover[phrom] > prb_cover[elite] - PRECISION and set_part[phrom] < set_part[elite]) elite = phrom;
		}
	}

	//evolutionary algorithm
	void evolutionary_algorithm()
	{
		double sample_fitness_best(0), prb_cover_best(0.0), set_part_best(1.0);
		short class_fitness_best(-1);
		Reproduction reproduce;
		population_generation();
		generation = 0;
		terminal_generation = __max(tr_size, minimum_generations);
		//start evolutionary algorithm
		do
		{
			generation++;
			fitness_evaluation();
			if (class_fitness[elite] > class_fitness_best)
			{
				class_fitness_best = class_fitness[elite];
				sample_fitness_best = sample_fitness[elite];
				prb_cover_best = prb_cover[elite];
				set_part_best = set_part[elite];
				terminal_generation += 4;
			}
			else if (class_fitness[elite] == class_fitness_best and sample_fitness[elite] > sample_fitness_best)
			{
				sample_fitness_best = sample_fitness[elite];
				prb_cover_best = prb_cover[elite];
				set_part_best = set_part[elite];
				terminal_generation += 3;
			}
			else if (class_fitness[elite] == class_fitness_best and sample_fitness[elite] > sample_fitness_best - PRECISION and prb_cover[elite] > prb_cover_best)
			{
				prb_cover_best = prb_cover[elite];
				set_part_best = set_part[elite];
				terminal_generation += 2;
			}
			else if (class_fitness[elite] == class_fitness_best and sample_fitness[elite] > sample_fitness_best - PRECISION and prb_cover[elite] > prb_cover_best - PRECISION and set_part[elite] < set_part_best)
			{
				set_part_best = set_part[elite];
				terminal_generation++;
			}
			reproduce.selection_scheme();
			reproduce.operator_selection();
		} while (generation < terminal_generation and class_fitness[elite] < ultimate_fitness);
	}
};

class Split {
private:
	// some data
	unsigned short spl;
	// some functions
public:
	//generating training and validation set
	void train_test_split(unsigned short ts_fold)
	{
		tr_size = ts_size = 0;
		for (unsigned short spl = 1; spl <= num_samples; spl++)
		{
			class_is[0][spl] = 1;			//number of classes for one sample initialized to 1
			if (subset[spl] == ts_fold)
				tss[++ts_size] = spl;
			else trs[++tr_size] = spl;
		}
	}

	//use alternative fold
	unsigned short alternative_fold(unsigned short fold)
	{
		if (fold == 1) return 2;
		return 1;
	}

	//generating folds
	void subsets(unsigned short num_folds)
	{
		unsigned short num_spl[MAXFOLDS][MAXCLASSES]{}, n_spl[MAXFOLDS]{}, fold;
		for (spl = 1; spl <= num_samples; spl++)
		{
			fold = 1 + rand() % num_folds;
			if (num_spl[fold][class_is[1][spl]] >= class_size[class_is[1][spl]] / float(num_folds))	//second priority constraint
				fold = alternative_fold(fold);
			if (n_spl[fold] >= num_samples / float(num_folds))	//first priority constraint
				fold = alternative_fold(fold);
			n_spl[fold]++;	//increment in number of samples in this fold
			num_spl[fold][class_is[1][spl]]++;	//increment in number of samples of this class in this fold
			subset[spl] = fold;	// fold assigned to element
		}
	}
};

//constructing a tree
void tree_construction()
{
	unsigned short node;
	for (node = 2 * num_features - 1; node > num_features - 1; node--)
	{
		tree[node][0] = 1;
		tree[node][1] = node + num_features;
	}
	for (; node > 0; node--)
	{
		tree[node][0] = 2;
		tree[node][1] = 2 * node;
		tree[node][2] = 2 * node + 1;
	}
}

//parse analyzer
unsigned short whatis(char* s)
{
	unsigned short w(0);
	if (strcmp(s, "number_of_features:") == 0) w = 1;
	else if (strcmp(s, "number_of_classes:") == 0) w = 2;
	else if (strcmp(s, "number_of_samples:") == 0) w = 3;
	else if (strcmp(s, "finished") == 0) w = 4;
	return w;
}

//parsing the dataset
void parser(string dataset)
{
	ifstream fin(dataset);        //object
	bool done(false), class_added;
	char instr[200];
	unsigned short spl, ftr, word, cls, classes_added(0);
	while (!done)
	{
		fin >> instr;
		word = whatis(instr);
		switch (word)
		{
		case 1: fin >> instr;
			num_features = atoi(instr);
			break;
		case 2: fin >> instr;
			num_classes = atoi(instr);
			for (cls = 1; cls <= num_classes; cls++)
				class_size[cls] = 0;
			break;
		case 3: fin >> instr;
			num_samples = atoi(instr);
			for (unsigned short i = 0; i <= num_features; i++)
				fin >> instr; //skip headings
			for (spl = 1; spl <= num_samples; spl++)
			{
				//fin >> composition[spl];
				for (ftr = 1; ftr <= num_features; ftr++)
				{
					fin >> feature[ftr][spl];
					feature_set[ftr][spl] = feature[ftr][spl];
				}
				fin >> instr;
                                class_added = false;
                                for (cls = 1; cls <= classes_added; cls++)
                                {
                                        if(strcmp(class_name[cls], instr) == 0)
                                        {
                                                class_is[0][spl] = 1;		//number of identical samples but belonging to different classes
						class_is[1][spl] = cls;
						class_size[class_is[1][spl]]++;
						class_added = true;
                                        }
                                }
                                if(not class_added)
                                {
                                        strcpy(class_name[++classes_added], instr);
                                        class_is[1][spl] = classes_added;
                                        class_size[class_is[1][spl]]++;
                                }
			}
			break;
		case 4: done = true; break;
		}
	}
}

//out put
void output(unsigned short itr, ofstream& fout, unsigned short fold)
{
	fout << "Training set result" << "\n";
	fout << "iteration = " << itr << "\n";
	fout << "fold = " << fold << "\n";
	fout << "Number of hierarchies = " << level << "\n";
	for (unsigned short phrom = 1; phrom <= level; phrom++)
	{
		fout << "Model Number = " << phrom << "\n";
		fout << "Model Fitness = " << model_sample_fitness[phrom] << "\n";
		fout << "Model Set Partition = " << model_set_part[phrom] << "\n";
		for (unsigned short id = 1; id <= num_classes; id++)
		{
			fout << "Model Maximum for class " << id << " = " << model_val_max[phrom][id] << "\n";
			fout << "Model Minimum for class " << id << " = " << model_val_min[phrom][id] << "\n";
			fout << "Model Mean for class " << id << " = " << model_val_mean[phrom][id] << "\n";
		}
		unsigned short gene;
		for (gene = 1; gene <= 3 * num_features - 1; gene++)
			fout << phromosome[phrom][gene] << "\n";
		for (gene = 2 * num_features - 1; gene >= num_features; gene--)
			fout << gene << " = " << phromosome[phrom][tree[gene][1]] << " x " << phromosome[phrom][gene] << "\n";
		for (; gene >= 1; gene--)
		{
			if (phromosome[phrom][gene] <= 4)
			{
				fout << gene << " = " << tree[gene][1];
				if (phromosome[phrom][gene] == 1) fout << " + ";
				else if (phromosome[phrom][gene] == 2) fout << " - ";
				else if (phromosome[phrom][gene] == 3) fout << " x ";
				else fout << " / ";
				fout << tree[gene][2] << "\n";
			}
			else
			{
				fout << gene << " = ";
				if (phromosome[phrom][gene] == 5) fout << "max(";
				else fout << "min(";
				fout << tree[gene][1] << ", " << tree[gene][2] << ")" << "\n";
			}
		}
		fout << "\n";
	}
	fout << "\n";
}

// results in percentage
unsigned short results()
{
	unsigned short correct_this_fold(0);
	for (unsigned short spl = 1; spl <= ts_size; spl++)
	{
		run_total++;
		if (class_found[tss[spl]] == class_is[1][tss[spl]])
		{
			correct_this_fold++;
			run_correct++;
		}
	}
	return correct_this_fold;
}

void x_sim_based_results(ofstream& fout, ofstream& xfout, ofstream& file_out)
{
        char fname[150];
        unsigned short spl, cls, cls_elected[MAXCLASSES]{}, num_correct(0), max_vote, num_elected;
        fout << "sample#" << "\t" << "Composition" << "\t";
        for (cls = 1; cls <= num_classes; cls++)
        {
                fout << "class";
                fout << class_name[cls];
                fout << "_votes";
                fout << "\t";
        }
        for (cls = 1; cls <= num_classes; cls++)
        {
                fout << "class";
                fout << class_name[cls];
                fout << "_wvotes";
                fout << "\t";
        }
        fout << "elected_class" << "\t" << "actual_class" << "\n";
        for (spl = 1; spl <= num_samples; spl++)
        {
                fout << spl << "\t";
                max_vote = num_elected = 0;
                for (cls = 1; cls <= num_classes; cls++)
                        fout << vote[cls][spl] << "\t";
                for (cls = 1; cls <= num_classes; cls++)
                        fout << weighted_vote[cls][spl] << "\t";
                for (cls = 1; cls <= num_classes; cls++)
                {
                        if (vote[cls][spl] > max_vote)
                        {
                                max_vote = vote[cls][spl];
                                num_elected = 1;
                                cls_elected[num_elected] = cls;
                        }
                        else if (vote[cls][spl] == max_vote)
                        {
                                num_elected++;
                                cls_elected[num_elected] = cls;
                        }
                }
                if (num_elected > 1)
                {
                        max_vote = 0;
                        for (cls = 1; cls <= num_elected; cls++)
                        {
                                if (weighted_vote[cls_elected[cls]][spl] > max_vote)
                                {
                                        max_vote = weighted_vote[cls][spl];
                                        cls_elected[1] = cls;
                                }
                        }
                }
                fout << class_name[cls_elected[1]] << "\t" << class_name[class_is[1][spl]] << "\n";
                if (class_is[1][spl] == cls_elected[1]) num_correct++;
        }
        fout << "number of correct elections: " << num_correct << "\n";
        fout << "total number of samples: " << num_samples << "\n";
        fout << "Accuracy of elections: " << num_correct / double(num_samples) * 100 << "\n";
        xfout << num_correct / double(num_samples) * 100;
        xfout << "\n";
}

string intToString(int t)
{
	string ch;

	ostringstream outs;
	outs << t;   // Convert value into a string.
	ch = outs.str();

	return ch;
}

int main(int argc, char *argv[])
{
	unsigned short itr, fold;
	double average;
	double start_time, end_time;
	string fname, oname, dataset;
	Evolution evo;
	srand(time(NULL));
	num_folds = 2;
	dataset = argv[1];
	oname = argv[2];
	fname =  oname + "_test_" + intToString(num_folds) + "_" + intToString(NITRS) + ".xls";
	ofstream f_out(fname.c_str());
	fname = oname + "_training_" + intToString(num_folds) + "_" + intToString(NITRS) + ".xls";
	ofstream fout(fname.c_str());
	fname = oname + "_ensemble_" + intToString(num_folds) + "_" + intToString(NITRS) + ".xls";
	ofstream xfout(fname.c_str());
	fname = oname + "_levels_" + intToString(num_folds) + "_" + intToString(NITRS) + ".xls";
	ofstream file_out(fname.c_str());      //object
	parser(dataset);		//reading the dataset
	tree_construction();	//construction of tree for model representation
	chrom_length = 3 * num_features - 1;
	minimum_generations = __min(chrom_length * 2, 30);
	ultimate_fitness = num_classes - 1;
	xfout << "Average Accuracy" << endl;
	start_time = clock();
	for (unsigned short spl = 1; spl <= num_samples; spl++)
	{
		for (unsigned short cls = 1; cls <= num_classes; cls++)
		{
			vote[cls][spl] = 0;
			weighted_vote[cls][spl] = 0;
		}
	}
	for (itr = 1; itr <= NITRS; itr++)
	{
		Split splt;
		//creating training/test sets
		splt.subsets(num_folds);
		run_correct = run_total = 0;
		for (fold = 1; fold <= num_folds; fold++)
		{
			level = 0;
			splt.train_test_split(fold);				//generating training set
			//Resolving the conflicts if they exist
			Conflict_Resolution cr{};
			bool data_resolve;
			unsigned short spl1, spl2;
			data_resolve = cr.similarity_index_check(&spl1, &spl2);
			if (data_resolve)
			{
				do
				{
					cr.conflict_resolution(spl1, spl2);
					data_resolve = cr.similarity_index_check(&spl1, &spl2);
				} while (data_resolve);
			}
			Classification clf;
			do
			{
				evo.evolutionary_algorithm();
				clf.set_partition();
			} while (class_fitness[elite] < ultimate_fitness);
			output(itr, fout, fold); //training set results
			clf.test_set();
			unsigned short num_correct = results();
			file_out << "sample#" << "\t" << "predicted_class" << "\t" << "actual_class" << "\t" << "decision_level" << "\n";
			for (unsigned short spl = 1; spl <= ts_size; spl++)
			{
				fout << tss[spl] << "\t" << class_found[tss[spl]] << "\t" << class_is[1][tss[spl]] << "\n";
				file_out << tss[spl] << "\t" << class_found[tss[spl]] << "\t" << class_is[1][tss[spl]] << "\t" << test_level[tss[spl]] << "\n";
			}
			fout << "Accuracy in this fold = " << num_correct / double(ts_size) << "\n";
		}
	}
	x_sim_based_results(f_out, xfout, file_out);
	end_time = clock();
	xfout << "\n" << "\n" << "\n" << endl;
	xfout << "Execution Time = " << (double)(end_time - start_time) / CLOCKS_PER_SEC << "\n";
	return 0;
}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
