function [ out ] = MB8( params,pnames,learning_data,post_learning_data,simulation,prior)
    ID=1; AGENT = 2; SESSION = 3; TRIAL=4 ;STATE=5; P1 = 6; P2=7; MAG=8; VAL = 9; INF = 10; CHOICE=11; OUTCOME=12; CF_OUTCOME=13;CONDITION=14;
    
    nb_states   = max(learning_data(:,STATE));

    beta        = params(1);                                                           
    alpha_rpe   = params(2);
    alpha_sape  = params(3);
    persev      = params(4);
    alpha_inv   = 0.1;%params(5);
    
    last_choice = zeros(nb_states); 
    
    Q = zeros(nb_states,2);
    Qd = Q;
    P = exp(beta*Q)./sum(exp(beta*Q),2);

    Qs = Q; %value function for the demonstrator
    
    if(simulation)
        lik_all = zeros(length(learning_data),1);
        Q_all = zeros(length(learning_data),2);
    end
        
    p=0;
    if(prior)
        p = parameter_priors(params,pnames);
    end

    lik=0;

    for i = 1:length(learning_data(:,TRIAL))

        if (~isnan(learning_data(i,AGENT)) && ~isnan(learning_data(i,CHOICE)))
            
            Qd(learning_data(i,STATE),:) = Q(learning_data(i,STATE),:);
            
            if(learning_data(i,CONDITION)~=0)
                %use the demonstrator model for bias
                [~,dem] =max(Qs(learning_data(i,STATE),:));
                
                sAPE = 1-Qd(learning_data(i,STATE),dem);
                Qd(learning_data(i,STATE),dem) = Qd(learning_data(i,STATE),dem) + alpha_sape * sAPE;
                
            end           
            
            if(last_choice(learning_data(i,STATE))>0)
                Qd(learning_data(i,STATE),last_choice(learning_data(i,STATE))) = Qd(learning_data(i,STATE),last_choice(learning_data(i,STATE))) + persev;
            end
            
            P(learning_data(i,STATE),:) = exp(beta*Qd(learning_data(i,STATE),:))./sum(exp(beta*Qd(learning_data(i,STATE),:)));
            
            if(simulation)
                lik = lik + log(P(learning_data(i,STATE),learning_data(i,CHOICE)));
                lik_all(i)=lik;
                Q_all(i,:) = Q(learning_data(i,STATE),:);
                
                [choice,out,cf_out] = one_step(P(learning_data(i,STATE),:),learning_data(i,MAG),learning_data(i,VAL),learning_data(i,INF),learning_data(i,P1),learning_data(i,P2));
                
                learning_data(i,CHOICE)     = choice;
                learning_data(i,OUTCOME)    = out;
                learning_data(i,CF_OUTCOME) = cf_out;

            else
                
                lik = lik + log(P(learning_data(i,STATE),learning_data(i,CHOICE)));%compute likelihood
                
            end

            %private learning
            RPE =  learning_data(i,OUTCOME) - Q(learning_data(i,STATE),learning_data(i,CHOICE));
            Q(learning_data(i,STATE),learning_data(i,CHOICE)) = Q(learning_data(i,STATE),learning_data(i,CHOICE)) + alpha_rpe * RPE;

	    %symmetric updating
            RPE2 =  -learning_data(i,OUTCOME) - Q(learning_data(i,STATE),3-learning_data(i,CHOICE));
            Q(learning_data(i,STATE),3-learning_data(i,CHOICE)) = Q(learning_data(i,STATE),3-learning_data(i,CHOICE)) + alpha_rpe * RPE2;

            last_choice(learning_data(i,STATE))=learning_data(i,CHOICE);

        elseif(isnan(learning_data(i,AGENT)) && ~isnan(learning_data(i,CHOICE)))
            %update the demonstrator's model
            delta =  1 - Qs(learning_data(i,STATE),learning_data(i,CHOICE));
            Qs(learning_data(i,STATE),learning_data(i,CHOICE)) = Qs(learning_data(i,STATE),learning_data(i,CHOICE)) + alpha_inv * delta;
            
            delta2 =  -1 - Qs(learning_data(i,STATE),3-learning_data(i,CHOICE));
            Qs(learning_data(i,STATE),3-learning_data(i,CHOICE)) = Qs(learning_data(i,STATE),3-learning_data(i,CHOICE)) + alpha_inv * delta2;
            
            if(simulation)
                Q_all(i,:) = Qs(learning_data(i,STATE),:);
            end
        end

    end

    l= -lik;
    out=p+l;
    out=min(out,realmax);
    if(simulation)
        out = struct();
        out.sim_data=learning_data;
        out.pred_data = [lik_all  Q_all];
    end

end
