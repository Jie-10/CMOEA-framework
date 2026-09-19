function [Population,Fitness] = EnviromentSelect3(Population,N,MinAngle,W)
% Diversity-oriented constrained environmental selection

    %% Basic data
    Obj  = Population.objs;
    Con  = Population.cons;

    NumQ = length(Population);
    NW   = size(W,1);

    %% Calculate cosine similarity between solutions and reference vectors
    CosQW = 1 - pdist2(Obj,W,'cosine');

    % Avoid numerical errors
    CosQW = min(max(CosQW,-1),1);
    CosQW(isnan(CosQW)) = 0;

    %% Determine solutions contained in each subspace
    CosThreshold = cos(MinAngle);

    InSubspaceQ = CosQW >= CosThreshold;

    %% Calculate constrained fitness
    FitnessAll = CalFitness(Obj,Con,0);

    %% Environmental selection
    selectedIndex   = zeros(1,N);
    selectedFitness = zeros(1,N);

    selected      = false(1,NumQ);
    selectedCount = 0;

    for i = 1:NW

        if selectedCount >= N
            break;
        end

        %% Remaining solutions
        R = find(~selected);

        if isempty(R)
            break;
        end

        %% Solutions located in current subspace
        Ti = find(InSubspaceQ(:,i)' & ~selected);

        if isempty(Ti)

            %% Empty subspace:
            % Select the remaining solution closest to
            % the current reference vector
            [~,bestLocal] = max(CosQW(R,i));
            x = R(bestLocal);

        else

            %% Non-empty subspace:
            % Select the solution with the best constrained fitness
            [~,bestLocal] = min(FitnessAll(Ti));
            x = Ti(bestLocal);

        end

        %% Save selected solution
        selectedCount = selectedCount + 1;

        selectedIndex(selectedCount)   = x;
        selectedFitness(selectedCount) = FitnessAll(x);

        selected(x) = true;
    end

    if selectedCount < N

        R = find(~selected);

        if ~isempty(R)

            %% Sort remaining solutions according to fitness
            [~,Rank] = sort(FitnessAll(R),'ascend');

            %% Number of solutions still required
            Need = min(N-selectedCount,length(R));

            %% Select the best remaining solutions
            Add = R(Rank(1:Need));

            selectedIndex(selectedCount+1:selectedCount+Need) = Add;

            selectedFitness(selectedCount+1:selectedCount+Need) = ...
                FitnessAll(Add);

            selectedCount = selectedCount + Need;
        end
    end

    %% Output
    selectedIndex   = selectedIndex(1:selectedCount);
    selectedFitness = selectedFitness(1:selectedCount);

    Population = Population(selectedIndex);
    Fitness    = selectedFitness;

end