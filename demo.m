clear
clc

currentFilePath = mfilename('fullpath');
[currentDirectory, ~, ~] = fileparts(currentFilePath);
addpath(genpath(currentDirectory));

[parentpath,~]=fileparts(currentFilePath)
[parentpath,~]=fileparts(parentpath)

addpath(genpath(parentpath));


% databases
databases = ["Scene-15","3sources", "ORL", "MSRC_V1", "BBCsport", "COIL20_mv", "Caltech101-7","Handwritten","Caltech101-20"];

% repeat times
rep_times = 30;

% parameters
alpha_set = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 0.8, 1, 2, 5, 8, 10];
beta_set =  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.5, 0.8, 1, 2, 5, 8, 10];

alpha_len = length(alpha_set);
beta_len = length(beta_set);

eta = 0.5;
% main process
for data_index = 2:2%length(databases)
    dataname = databases(data_index);
    load(dataname)
    disp(["current processing:" dataname])
    
    num_sample = size(X{1},1);
    num_cluster = max(unique(Y));
    
    % data normalize
    view_num = size(X, 2); 
    for v_index=1:view_num
        tX = X{v_index};
        tX = tX./repmat(sqrt(sum(tX.^2, 2)),1,size(tX,2));
        X{v_index} = tX;
    end
    
    if min(unique(Y)) == 0
        num_cluster = num_cluster + 1;
    end
    
  
    % results
    Cuv_ACC_array = cell(alpha_len, beta_len);
    Cuv_NMI_array = cell(alpha_len, beta_len);
    Cuv_FSCORE_array = cell(alpha_len, beta_len);
    Cuv_PRECISION_array = cell(alpha_len, beta_len);
    Cuv_ARI_array = cell(alpha_len, beta_len);
    Cuv_RI_array = cell(alpha_len, beta_len);
    Cuv_PURITY_array = cell(alpha_len, beta_len);
    Cuv_RECALL_array = cell(alpha_len, beta_len);
    Cuv_array = cell(alpha_len, beta_len);
    LAMBDA_array = cell(alpha_len, beta_len);
    


    C_ACC_array = cell(alpha_len, beta_len);
    C_NMI_array = cell(alpha_len, beta_len);
    C_FSCORE_array = cell(alpha_len, beta_len);
    C_PRECISION_array = cell(alpha_len, beta_len);
    C_ARI_array = cell(alpha_len, beta_len);
    C_RI_array = cell(alpha_len, beta_len);
    C_PURITY_array = cell(alpha_len, beta_len);
    C_RECALL_array = cell(alpha_len, beta_len);
    C_array = cell(alpha_len, beta_len);

    for i = 3:length(alpha_set)
        for j = 1:length(beta_set)
            [C_uv, Z_uv, C, Z, lambda, C_uv_residual, Z_uv_residual] = mvsc2gf(X, alpha_set(i), beta_set(j), eta);

            for rep_ind = 1:rep_times
                [acc,nmi,F,precision,ari,ri,purity,recall] = get_measurements(C_uv, num_cluster,Y);
    
                acc_array(rep_ind) = acc;
                nmi_array(rep_ind) = nmi;
                fscore_array(rep_ind) = F;
                precision_array(rep_ind) = precision;
                ari_array(rep_ind) = ari;
                ri_array(rep_ind) = ri;
                purity_array(rep_ind) = purity;
                recall_array(rep_ind) = recall;
            end

            Cuv_ACC_array{i,j} = acc_array
            Cuv_NMI_array{i,j} = nmi_array;
            Cuv_FSCORE_array{i,j} = fscore_array;
            Cuv_PRECISION_array{i,j} = precision_array;
            Cuv_ARI_array{i,j} = ari_array;
            Cuv_RI_array{i,j} = ri_array;
            Cuv_PURITY_array{i,j} = purity_array;
            Cuv_RECALL_array{i,j} = recall_array;
            LAMBDA_array{i,j} = lambda;
            Cuv_array{i,j} = C_uv;

            C_fusion = zeros(num_sample,num_sample);
            for fun_ind=1:view_num
                C_fusion = C_fusion + lambda(fun_ind)*C{fun_ind};
            end

            for rep_ind = 1:rep_times
                [acc,nmi,F,precision,ari,ri,purity,recall] = get_measurements(C_fusion, num_cluster,Y);
    
                acc_array(rep_ind) = acc;
                nmi_array(rep_ind) = nmi;
                fscore_array(rep_ind) = F;
                precision_array(rep_ind) = precision;
                ari_array(rep_ind) = ari;
                ri_array(rep_ind) = ri;
                purity_array(rep_ind) = purity;
                recall_array(rep_ind) = recall;
            end
            C_ACC_array{i,j} = acc_array;
            C_NMI_array{i,j} = nmi_array;
            C_FSCORE_array{i,j} = fscore_array;
            C_PRECISION_array{i,j} = precision_array;
            C_ARI_array{i,j} = ari_array;
            C_RI_array{i,j} = ri_array;
            C_PURITY_array{i,j} = purity_array;
            C_RECALL_array{i,j} = recall_array;

        end
    end
    

    file = databases(data_index) + "_Cuv_enhanced_results_ablation";
    save(file, "Cuv_ACC_array", "Cuv_NMI_array", "Cuv_FSCORE_array", "Cuv_PRECISION_array","Cuv_ARI_array", ...
        "Cuv_RI_array","Cuv_PURITY_array","Cuv_RECALL_array","LAMBDA_array","Cuv_array")

    file = databases(data_index) + "_C_enhanced_results_ablation";
    save(file, "C_ACC_array", "C_NMI_array", "C_FSCORE_array", "C_PRECISION_array","C_ARI_array", ...
        "C_RI_array","C_PURITY_array","C_RECALL_array")
end